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

#--------------------------------------------------------------------------------------------------------
# Post-hoc fairness evaluation across the AC-feasible switch configurations.
#
# This is purely an evaluation — no optimization, no JuMP, no weight decisions, no Jacobian.
# For each AC-feasible configuration from `rounding_algorithm_brute_force.jl`:
#   1. Fix the switches and run the rounded integer MLD → per-load `pshed` (length N).
#   2. Evaluate every fairness function in `src/implementation/other_fair_funcs.jl` (and the
#      Palma ratio from `src/implementation/load_shed_as_parameter.jl`) directly on that pshed.
#   3. Save per-config: pshed (per load) and the fairness values.
#   4. Identify all configs that tie at the optimum for each fairness function.
#
# Single period: with the topology fixed, there is nothing to couple periods over (no
# Taylor-series link to a lower-level Jacobian), so duplicating the same shed across periods
# would just multiply every metric by a constant. Multiperiod becomes meaningful again when
# the bilevel iteration has a gradient update tying weights across periods.
#--------------------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "rounding_algorithm_brute_force.jl"))

outdir = joinpath(@__DIR__, "..", "results", string(Dates.today()), "brute_force_upper_level")
mkpath(outdir)

const TIE_TOL_REL = 1e-4
const TIE_TOL_ABS = 1e-7

const FAIR_FUNCS = ["efficient", "jain", "min_max", "proportional",
                    "equality_min", "palma"]

const FAIR_DIRECTIONS = Dict(
    "efficient"     => :min,
    "jain"          => :max,
    "min_max"       => :min,
    "proportional"  => :max,
    "equality_min"  => :min,
    "palma"         => :min,
)

#--------------------------------------------------------------------------------------------------------
# Single-period fairness formulas — same as the JuMP objectives in other_fair_funcs.jl, evaluated
# directly on the fixed pshed vector. Proportional drops the ε offset (per request).
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

eval_equality_min(p) = sum(p) + (maximum(p) - minimum(p))

eval_palma(p) = FairLoadDelivery.palma_ratio(p)

#--------------------------------------------------------------------------------------------------------
# Setup
#--------------------------------------------------------------------------------------------------------

load_ids_sorted = sort(parse.(Int, collect(keys(math["load"]))))
n               = length(load_ids_sorted)
pd_per_load     = Float64[sum(math["load"][string(lid)]["pd"]) for lid in load_ids_sorted]

config_pshed = Dict{String, Vector{Float64}}()         # per-load pshed (length N)
config_obj   = Dict{String, Dict{String, Float64}}()
config_mld_status = Dict{String, Symbol}()

#--------------------------------------------------------------------------------------------------------
# Main loop — fix switches, run rounded integer MLD, evaluate each fairness function.
#--------------------------------------------------------------------------------------------------------

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

    mld_res = FairLoadDelivery.solve_mc_mld_shed_random_round_integer(trial, Gurobi.Optimizer)
    tstat = mld_res["termination_status"]
    if tstat ∉ (MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
        @warn "config $cfg: rounded integer MLD failed ($tstat) — skipping"
        continue
    end

    pshed = Float64[sum(mld_res["solution"]["load"][string(lid)]["pshed"])
                    for lid in load_ids_sorted]
    config_pshed[cfg]      = pshed
    config_mld_status[cfg] = Symbol(tstat)

    config_obj[cfg] = Dict(
        "efficient"     => eval_efficient(pshed),
        "jain"          => eval_jain(pshed),
        "min_max"       => eval_min_max(pshed),
        "proportional"  => eval_proportional(pshed, pd_per_load),
        "equality_min"  => eval_equality_min(pshed),
        "palma"         => eval_palma(pshed),
    )

    @info "config $cfg: total_shed=$(round(sum(pshed), digits=3))  " *
          "jain=$(round(config_obj[cfg]["jain"], digits=4))  " *
          "min_max=$(round(config_obj[cfg]["min_max"], digits=3))  " *
          "prop=$(round(config_obj[cfg]["proportional"], digits=3))  " *
          "palma=$(round(config_obj[cfg]["palma"], digits=3))"
end

#--------------------------------------------------------------------------------------------------------
# Save outputs
#--------------------------------------------------------------------------------------------------------

successful_configs = sort(collect(keys(config_obj)))

# 1) Wide summary: one row per config with totals + every fairness value.
summary_df = DataFrame(
    config_bits       = String[],
    total_pshed       = Float64[],
    obj_efficient     = Float64[],
    obj_jain          = Float64[],
    obj_min_max       = Float64[],
    obj_proportional  = Float64[],
    obj_equality_min  = Float64[],
    obj_palma         = Float64[],
)
for cfg in successful_configs
    push!(summary_df, (cfg, sum(config_pshed[cfg]),
        config_obj[cfg]["efficient"], config_obj[cfg]["jain"],
        config_obj[cfg]["min_max"],   config_obj[cfg]["proportional"],
        config_obj[cfg]["equality_min"], config_obj[cfg]["palma"]))
end
CSV.write(joinpath(outdir, "summary.csv"), summary_df)

# 2) Per-load pshed (long format) — pshed from the rounded integer MLD per config.
pshed_long_df = DataFrame(config_bits=String[], load_id=Int[], pshed=Float64[])
for cfg in successful_configs
    for (i, lid) in enumerate(load_ids_sorted)
        push!(pshed_long_df, (cfg, lid, config_pshed[cfg][i]))
    end
end
CSV.write(joinpath(outdir, "pshed_per_config.csv"), pshed_long_df)

# 3) Topology equivalence groups — many of the 47 AC-feasible switch configs collapse to the
#    same per-load pshed vector (electrically identical operating point). Group configs by their
#    pshed signature so downstream tied-optima / plots can use one representative per group.
function pshed_signature(p::Vector{Float64}; digits::Int=6)
    return Tuple(round.(p; digits=digits))
end
sig_to_configs = Dict{Tuple, Vector{String}}()
for cfg in successful_configs
    sig = pshed_signature(config_pshed[cfg])
    push!(get!(sig_to_configs, sig, String[]), cfg)
end
unique_configs = sort([first(sort(group)) for group in values(sig_to_configs)])

equivalence_df = DataFrame(group_id=Int[], representative=String[], member=String[],
                           total_pshed=Float64[], n_in_group=Int[])
for (gid, group) in enumerate(sort(collect(values(sig_to_configs)); by=g -> first(sort(g))))
    rep = first(sort(group))
    total = sum(config_pshed[rep])
    for cfg in sort(group)
        push!(equivalence_df, (gid, rep, cfg, total, length(group)))
    end
end
CSV.write(joinpath(outdir, "topology_equivalence_groups.csv"), equivalence_df)

@info "Topology de-dup: $(length(successful_configs)) AC-feasible configs → " *
      "$(length(unique_configs)) distinct pshed signatures"

# 4) Tied-at-optimum: every config within tol of the optimum for each fairness function.
#    Operate on the de-duplicated representative set so a "tie" reflects a genuinely different
#    operating point, not a redundant switch labelling.
tied_df = DataFrame(fair_func=String[], config_bits=String[], obj=Float64[],
                    total_pshed=Float64[], direction=String[])
for ff in FAIR_FUNCS
    objs = Float64[config_obj[cfg][ff] for cfg in unique_configs]
    finite_objs = filter(isfinite, objs)
    isempty(finite_objs) && continue

    if FAIR_DIRECTIONS[ff] == :max
        best = maximum(finite_objs)
        tol  = max(TIE_TOL_ABS, abs(best) * TIE_TOL_REL)
        tied_idx = findall(o -> isfinite(o) && o >= best - tol, objs)
    else
        best = minimum(finite_objs)
        tol  = max(TIE_TOL_ABS, abs(best) * TIE_TOL_REL)
        tied_idx = findall(o -> isfinite(o) && o <= best + tol, objs)
    end

    for i in tied_idx
        cfg = unique_configs[i]
        push!(tied_df, (ff, cfg, objs[i], sum(config_pshed[cfg]),
                        string(FAIR_DIRECTIONS[ff])))
    end
end
CSV.write(joinpath(outdir, "tied_optima.csv"), tied_df)

println("\n=== Tied optima per fairness function ===")
for ff in FAIR_FUNCS
    n_tied = count(==(ff), tied_df.fair_func)
    @info "  $ff: $n_tied tied configuration(s)"
end

#--------------------------------------------------------------------------------------------------------
# Plots
#--------------------------------------------------------------------------------------------------------

cfg_labels = summary_df.config_bits

plt_eff   = bar(cfg_labels, summary_df.obj_efficient;
                title="Efficient: min Σ pshed",            xrotation=90, legend=false)
plt_jain  = bar(cfg_labels, summary_df.obj_jain;
                title="Jain: max",                         xrotation=90, legend=false)
plt_mm    = bar(cfg_labels, summary_df.obj_min_max;
                title="Min-Max: min",                      xrotation=90, legend=false)
plt_prop  = bar(cfg_labels, summary_df.obj_proportional;
                title="Proportional: max Σ log(served + 1)", xrotation=90, legend=false)
plt_eq    = bar(cfg_labels, summary_df.obj_equality_min;
                title="Equality-min: min",                 xrotation=90, legend=false)
plt_palma = bar(cfg_labels, summary_df.obj_palma;
                title="Palma: min",                        xrotation=90, legend=false)
fig = plot(plt_eff, plt_jain, plt_mm, plt_prop, plt_eq, plt_palma;
           layout=(3, 2), size=(1600, 1200),
           plot_title="Post-hoc fairness evaluation over $(length(successful_configs)) AC-feasible configs")
savefig(fig, joinpath(outdir, "brute_force_upper_level.svg"))
display(fig)

#--------------------------------------------------------------------------------------------------------
# Jain vs Min-Max comparison on the de-duplicated representative configs
#--------------------------------------------------------------------------------------------------------

unique_summary = filter(row -> row.config_bits in unique_configs, summary_df)
sort!(unique_summary, :total_pshed)

# Scatter (min_max, jain) — each point a distinct operating topology, sized by total shed,
# annotated with the representative config bits. Min-Max best is bottom-left of x; Jain best
# is top of y. If they disagree, you'll see them at different points on the cloud.
size_scale = 6 .+ 18 .* (unique_summary.total_pshed .- minimum(unique_summary.total_pshed)) ./
                       max(maximum(unique_summary.total_pshed) - minimum(unique_summary.total_pshed), 1.0)

best_mm_idx   = argmin(unique_summary.obj_min_max)
best_jain_idx = argmax(unique_summary.obj_jain)

scatter_jain_mm = scatter(unique_summary.obj_min_max, unique_summary.obj_jain;
    xlabel = "Min-Max objective (lower → fairer worst case)",
    ylabel = "Jain index (higher → more uniform shed)",
    title  = "Jain vs Min-Max across $(length(unique_configs)) distinct operating points",
    markersize = size_scale,
    markeralpha = 0.6,
    color  = :steelblue,
    label  = "config (size ∝ total shed)",
    legend = :bottomright)
for r in eachrow(unique_summary)
    annotate!(scatter_jain_mm, r.obj_min_max, r.obj_jain,
              text(r.config_bits, 7, :black, :bottom))
end
scatter!(scatter_jain_mm,
    [unique_summary.obj_min_max[best_mm_idx]], [unique_summary.obj_jain[best_mm_idx]];
    markersize = 12, markershape = :star5, color = :red,
    label = "Min-Max best (cfg $(unique_summary.config_bits[best_mm_idx]), shed=$(Int(unique_summary.total_pshed[best_mm_idx])))")
scatter!(scatter_jain_mm,
    [unique_summary.obj_min_max[best_jain_idx]], [unique_summary.obj_jain[best_jain_idx]];
    markersize = 12, markershape = :diamond, color = :green,
    label = "Jain best (cfg $(unique_summary.config_bits[best_jain_idx]), shed=$(Int(unique_summary.total_pshed[best_jain_idx])))")
savefig(scatter_jain_mm, joinpath(outdir, "jain_vs_min_max.svg"))
display(scatter_jain_mm)

# Side-by-side: total shed vs each metric, so the "min_max sheds less than Jain" pattern is direct.
plt_shed_mm   = scatter(unique_summary.obj_min_max,   unique_summary.total_pshed;
    xlabel="Min-Max",  ylabel="Total shed", title="Total shed vs Min-Max",
    markersize=8, color=:steelblue, label=false)
scatter!(plt_shed_mm, [unique_summary.obj_min_max[best_mm_idx]],
    [unique_summary.total_pshed[best_mm_idx]];
    markersize=12, markershape=:star5, color=:red,
    label="Min-Max best")

plt_shed_jain = scatter(unique_summary.obj_jain,      unique_summary.total_pshed;
    xlabel="Jain index", ylabel="Total shed", title="Total shed vs Jain",
    markersize=8, color=:steelblue, label=false)
scatter!(plt_shed_jain, [unique_summary.obj_jain[best_jain_idx]],
    [unique_summary.total_pshed[best_jain_idx]];
    markersize=12, markershape=:diamond, color=:green,
    label="Jain best")

fig_shed = plot(plt_shed_mm, plt_shed_jain; layout=(1, 2), size=(1400, 500),
    plot_title="Total load shed vs each fairness metric (de-dup'd configs)")
savefig(fig_shed, joinpath(outdir, "shed_vs_jain_min_max.svg"))
display(fig_shed)

# Also dump the head-to-head table so the comparison is in CSV form.
jain_vs_mm_df = select(unique_summary, :config_bits, :total_pshed, :obj_min_max, :obj_jain)
jain_vs_mm_df.delta_shed_to_mm_best   = jain_vs_mm_df.total_pshed .- minimum(jain_vs_mm_df.total_pshed[unique_summary.obj_min_max .== minimum(unique_summary.obj_min_max)])
jain_vs_mm_df.delta_shed_to_jain_best = jain_vs_mm_df.total_pshed .- jain_vs_mm_df.total_pshed[best_jain_idx]
CSV.write(joinpath(outdir, "jain_vs_min_max.csv"), jain_vs_mm_df)

@info "Jain vs Min-Max best topologies:"
@info "  Min-Max best: cfg=$(unique_summary.config_bits[best_mm_idx]) " *
      "min_max=$(unique_summary.obj_min_max[best_mm_idx]) " *
      "jain=$(round(unique_summary.obj_jain[best_mm_idx], digits=4)) " *
      "total_shed=$(unique_summary.total_pshed[best_mm_idx])"
@info "  Jain best:    cfg=$(unique_summary.config_bits[best_jain_idx]) " *
      "min_max=$(unique_summary.obj_min_max[best_jain_idx]) " *
      "jain=$(round(unique_summary.obj_jain[best_jain_idx], digits=4)) " *
      "total_shed=$(unique_summary.total_pshed[best_jain_idx])"
