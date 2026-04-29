using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS
using JuMP
using LinearAlgebra, SparseArrays
using DataFrames
using CSV
using Dates
using Plots
using StatsPlots
using Statistics
using DiffOpt

#--------------------------------------------------------------------------------------------------------
# Two-period bilevel iteration over the AC-feasible switch configurations.
#
# For each AC-feasible topology from `rounding_algorithm_brute_force.jl`:
#
#   PERIOD 1 — implicit-diff MLD with the topology pinned, then the upper level:
#     a. Instantiate `build_mc_mld_shedding_implicit_diff(pm; fixed_topology=true)` so the
#        rounded-switch constraint is added inside the DiffOpt model. Solve once and compute
#        the full forward Jacobian dpshed/dw via DiffOpt.
#     b. Run each upper-level fairness function (other_fair_funcs.jl + lin_palma_reformulated)
#        with `(dpshed_dw, pshed_1, weights_init, pd)` → `(pshed_predicted, weights_new)`.
#     c. Record the fairness values on the upper-level predicted pshed.
#
#   PERIOD 2 — regular rounded integer MLD with the new weights and the same fixed topology:
#     d. Write `weights_new` into `math["load"][·]["weight"]`, then solve
#        `solve_mc_mld_shed_random_round_integer` → integer pshed_2.
#     e. Record the fairness values on pshed_2.
#
# Single-period upper level (n_loads = N, no peak_time_costs) — multiperiod becomes meaningful
# again only when the bilevel is iterated further with a Taylor-series-coupled lower level.
#--------------------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "rounding_algorithm_brute_force.jl"))

outdir = joinpath(@__DIR__, "..", "results", string(Dates.today()), "bilevel_two_period")
mkpath(outdir)

const FAIR_FUNCS = ["efficient"]

#--------------------------------------------------------------------------------------------------------
# Direct fairness evaluators (single period) — same formulas as `brute_force_upper_level.jl`.
# Proportional drops the ε offset (per request: log(0) = -Inf for fully shed loads).
#--------------------------------------------------------------------------------------------------------

eval_efficient(p)     = sum(p)
eval_min_max(p)       = maximum(p)

function eval_jain(p)
    sp  = sum(p)
    sp2 = sum(pi^2 for pi in p)
    return sp2 > 1e-12 ? (sp^2) / (length(p) * sp2) : 0.0
end

function eval_proportional(p, pd)
    # log(served + 1): finite floor of 0 for fully-shed loads (instead of -Inf), positive
    # contribution for any partial/full service. Avoids the corner-solution failure where
    # the integer-MLD's all-or-nothing block shedding makes log(served) = -Inf for every config.
    val = 0.0
    for i in eachindex(p)
        served = pd[i] - p[i]
        val += log(max(served, 0.0) + 1.0)
    end
    return val
end

function eval_equality_min(p)
    pmax = maximum(p)
    return pmax + sum((pi - pmax)^2 for pi in p)
end

eval_palma(p) = FairLoadDelivery.palma_ratio(p)

#--------------------------------------------------------------------------------------------------------
# Setup
#--------------------------------------------------------------------------------------------------------

load_ids_sorted = sort(parse.(Int, collect(keys(math["load"]))))
n               = length(load_ids_sorted)
pd_per_load     = Float64[sum(math["load"][string(lid)]["pd"])         for lid in load_ids_sorted]
weights_init    = Float64[Float64(math["load"][string(lid)]["weight"]) for lid in load_ids_sorted]

ε = 1e-7

# Long-format outputs
fairness_long = DataFrame(
    config_bits     = String[],
    fair_func       = String[],
    period          = Int[],
    total_pshed     = Float64[],
    obj_efficient   = Float64[],
    obj_jain        = Float64[],
    obj_min_max     = Float64[],
    obj_proportional = Float64[],
    obj_equality_min = Float64[],
    obj_palma       = Float64[],
)
pshed_long = DataFrame(
    config_bits=String[], fair_func=String[], period=Int[],
    load_id=Int[], pshed=Float64[],
)
weights_long = DataFrame(
    config_bits=String[], fair_func=String[],
    load_id=Int[], weight_new=Float64[],
)

# Helper: append fairness/pshed rows
function record_fairness_row!(cfg, ff, period, pshed, pd)
    push!(fairness_long, (cfg, ff, period, sum(pshed),
        eval_efficient(pshed),
        eval_jain(pshed),
        eval_min_max(pshed),
        eval_proportional(pshed, pd),
        eval_equality_min(pshed),
        eval_palma(pshed)))
end

#--------------------------------------------------------------------------------------------------------
# De-duplicate AC-feasible configs by their integer-MLD pshed signature, so the slow
# implicit-diff + DiffOpt pipeline only fires once per electrically distinct topology.
# Many of the 47 AC-feasible switch configurations collapse to the same per-load pshed
# vector (electrically identical operating point) — running 12 reps instead of 47 cuts
# the heavy work by ~75%.
#--------------------------------------------------------------------------------------------------------

function _pshed_signature(p::Vector{Float64}; digits::Int=6)
    return Tuple(round.(p; digits=digits))
end

ac_pshed_lookup = Dict{String, Vector{Float64}}()
for row in eachrow(ac_feasible_networks)
    cfg = row.config_bits
    states = Dict(sid => Float64(parse(Int, cfg[i]))
                  for (i, sid) in enumerate(sorted_switch_ids))
    trial = deepcopy(math)
    for (sid, s) in states
        trial["switch"][sid]["state"]        = s
        trial["switch"][sid]["status"]       = s
        trial["switch"][sid]["dispatchable"] = 1.0
    end
    for (i, lid) in enumerate(load_ids_sorted)
        trial["load"][string(lid)]["weight"] = weights_init[i]
    end
    mld_init = FairLoadDelivery.solve_mc_mld_shed_random_round_integer(trial, Gurobi.Optimizer)
    if mld_init["termination_status"] ∉ (MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
        @warn "config $cfg: initial integer MLD failed ($(mld_init["termination_status"])) — excluded from de-dup"
        continue
    end
    ac_pshed_lookup[cfg] = Float64[sum(mld_init["solution"]["load"][string(lid)]["pshed"])
                                   for lid in load_ids_sorted]
end

sig_to_configs = Dict{Tuple, Vector{String}}()
for (cfg, p) in ac_pshed_lookup
    push!(get!(sig_to_configs, _pshed_signature(p), String[]), cfg)
end
unique_configs = sort([first(sort(group)) for group in values(sig_to_configs)])

# Persist the equivalence groups so downstream rows keyed on `representative` can be
# expanded back to the full list of equivalent switch configurations.
equivalence_df = DataFrame(group_id=Int[], representative=String[], member=String[],
                           total_pshed=Float64[], n_in_group=Int[])
for (gid, group) in enumerate(sort(collect(values(sig_to_configs)); by=g -> first(sort(g))))
    rep = first(sort(group))
    total = sum(ac_pshed_lookup[rep])
    for cfg in sort(group)
        push!(equivalence_df, (gid, rep, cfg, total, length(group)))
    end
end
CSV.write(joinpath(outdir, "topology_equivalence_groups.csv"), equivalence_df)

@info "Bilevel de-dup: $(length(ac_pshed_lookup)) AC-feasible configs → " *
      "$(length(unique_configs)) distinct pshed signatures (running pipeline on reps only)"

#--------------------------------------------------------------------------------------------------------
# Main loop — period 1 (implicit-diff + upper level) → period 2 (regular MLD with new weights)
#--------------------------------------------------------------------------------------------------------

for cfg in unique_configs
    states = Dict(sid => Float64(parse(Int, cfg[i]))
                  for (i, sid) in enumerate(sorted_switch_ids))

    trial = deepcopy(math)
    for (sid, s) in states
        trial["switch"][sid]["state"]        = s
        trial["switch"][sid]["status"]       = s
        trial["switch"][sid]["dispatchable"] = 1.0
    end
    # Reset weights to the initial values for the period-1 lower level
    for (i, lid) in enumerate(load_ids_sorted)
        trial["load"][string(lid)]["weight"] = weights_init[i]
    end

    #=========== PERIOD 1: implicit-diff MLD with fixed topology + DiffOpt Jacobian ===========#
    build_fn = pm -> build_mc_mld_shedding_implicit_diff(pm; fixed_topology=true)
    mld_paramed = instantiate_mc_model(trial, LinDist3FlowPowerModel, build_fn;
        ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])

    local dpshed_dw, pshed_1_raw, ll_pshed_ids, ll_weight_vals, ll_weight_ids
    try
        dpshed_dw, pshed_1_raw, ll_pshed_ids, ll_weight_vals, ll_weight_ids, _ =
            FairLoadDelivery.diff_forward_full_jacobian(mld_paramed.model, weights_init)
    catch e
        @warn "config $cfg: DiffOpt forward Jacobian failed — $e — skipping"
        continue
    end

    # Reorder lower-level outputs to align with load_ids_sorted
    pshed_perm  = [findfirst(==(lid), ll_pshed_ids)  for lid in load_ids_sorted]
    weight_perm = [findfirst(==(lid), ll_weight_ids) for lid in load_ids_sorted]
    if any(isnothing, pshed_perm) || any(isnothing, weight_perm)
        @warn "config $cfg: load id mismatch between math and lower-level model — skipping"
        continue
    end
    pshed_1            = pshed_1_raw[pshed_perm]
    dpshed_dw_ordered  = dpshed_dw[pshed_perm, weight_perm]
    weights_init_ord   = ll_weight_vals[weight_perm]

    # Clamp pshed_1 strictly inside [ε, pd-ε] for proportional/palma feasibility
    pshed_1_clamped = Float64[clamp(pshed_1[i], ε, max(pd_per_load[i] - ε, ε)) for i in 1:n]

    #=========== UPPER LEVEL + PERIOD 2 PER FAIRNESS FUNCTION ===========#
    for ff in FAIR_FUNCS
        local pshed_pred, weights_new, status_upper
        try
            if ff == "efficient"
                pshed_pred, weights_new, status_upper = efficient_load_shed(
                    dpshed_dw_ordered, pshed_1, weights_init_ord;
                    n_loads=n)
            elseif ff == "jain"
                pshed_pred, weights_new, status_upper = jains_fairness_index(
                    dpshed_dw_ordered, pshed_1, weights_init_ord;
                    n_loads=n, pd=pd_per_load, alpha=1.0, reg=0.0)
            elseif ff == "min_max"
                pshed_pred, weights_new, status_upper = min_max_load_shed(
                    dpshed_dw_ordered, pshed_1, weights_init_ord;
                    n_loads=n, pd=pd_per_load, alpha=1.0, reg=0.0)
            elseif ff == "proportional"
                pshed_pred, weights_new, status_upper = proportional_fairness_load_shed(
                    dpshed_dw_ordered, pshed_1_clamped, weights_init_ord, pd_per_load;
                    n_loads=n, alpha=1.0, reg=0.0)
            elseif ff == "equality_min"
                pshed_pred, weights_new, status_upper = equality_min(
                    dpshed_dw_ordered, pshed_1, weights_init_ord;
                    n_loads=n)
            elseif ff == "palma"
                pshed_pred, weights_new, status_upper = lin_palma_reformulated(
                    dpshed_dw_ordered, pshed_1_clamped, weights_init_ord, pd_per_load;
                    n_loads=n, alpha=1.0, reg=0.0)
            end
        catch e
            @warn "config $cfg / $ff: upper level failed — $e — skipping"
            continue
        end

        pshed_p1 = collect(pshed_pred)
        record_fairness_row!(cfg, ff, 1, pshed_p1, pd_per_load)
        for (i, lid) in enumerate(load_ids_sorted)
            push!(pshed_long,   (cfg, ff, 1, lid, pshed_p1[i]))
            push!(weights_long, (cfg, ff,    lid, Float64(weights_new[i])))
        end

        #=========== PERIOD 2: regular rounded integer MLD with the new weights ===========#
        trial_p2 = deepcopy(trial)
        for (i, lid) in enumerate(load_ids_sorted)
            trial_p2["load"][string(lid)]["weight"] = Float64(weights_new[i])
        end
        mld_p2 = FairLoadDelivery.solve_mc_mld_shed_random_round_integer(trial_p2, Gurobi.Optimizer)
        if mld_p2["termination_status"] ∉ (MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
            @warn "config $cfg / $ff: period-2 MLD failed — $(mld_p2["termination_status"])"
            continue
        end
        pshed_p2 = Float64[sum(mld_p2["solution"]["load"][string(lid)]["pshed"])
                           for lid in load_ids_sorted]

        record_fairness_row!(cfg, ff, 2, pshed_p2, pd_per_load)
        for (i, lid) in enumerate(load_ids_sorted)
            push!(pshed_long, (cfg, ff, 2, lid, pshed_p2[i]))
        end

        @info "cfg $cfg / $ff: P1 shed=$(round(sum(pshed_p1), digits=3)) " *
              "P2 shed=$(round(sum(pshed_p2), digits=3))"
    end
end

#--------------------------------------------------------------------------------------------------------
# Save outputs
#--------------------------------------------------------------------------------------------------------

CSV.write(joinpath(outdir, "fairness_per_config_period.csv"), fairness_long)
CSV.write(joinpath(outdir, "pshed_per_config_period.csv"),    pshed_long)
CSV.write(joinpath(outdir, "weights_per_config_fairness.csv"), weights_long)

#--------------------------------------------------------------------------------------------------------
# Plots — period 1 vs period 2 totals, and a per-fairness panel
#--------------------------------------------------------------------------------------------------------

shed_summary = combine(groupby(fairness_long, [:fair_func, :period]),
    :total_pshed => mean => :mean_total_shed,
    :total_pshed => median => :median_total_shed,
    nrow => :n_configs)
CSV.write(joinpath(outdir, "shed_summary_per_fairness_period.csv"), shed_summary)

panels = []
for ff in FAIR_FUNCS
    sub = fairness_long[fairness_long.fair_func .== ff, :]
    p1  = sort(sub[sub.period .== 1, :], :config_bits)
    p2  = sort(sub[sub.period .== 2, :], :config_bits)
    plt = plot(title="$ff: P1 vs P2 total shed", xlabel="config", ylabel="Σ pshed",
               legend=:topright, xrotation=90)
    plot!(plt, p1.config_bits, p1.total_pshed; label="period 1 (predicted)", marker=:circle)
    plot!(plt, p2.config_bits, p2.total_pshed; label="period 2 (MLD w/ new weights)", marker=:square)
    push!(panels, plt)
end
fig = plot(panels...; layout=(4, 2), size=(1600, 1400),
           plot_title="Two-period bilevel: period 1 (predicted) vs period 2 (rounded MLD)")
savefig(fig, joinpath(outdir, "p1_vs_p2_total_shed.svg"))
display(fig)
