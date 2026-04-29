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

const FAIR_FUNCS = ["efficiency", "palma"]

const FAIR_DIRECTIONS = Dict(
    "efficient" => :min,
    "palma"     => :min,
)

#--------------------------------------------------------------------------------------------------------
# Single-period fairness formulas — same as the JuMP objectives in other_fair_funcs.jl, evaluated
# directly on the fixed pshed vector.
#--------------------------------------------------------------------------------------------------------

eval_efficient(p) = sum(p)
eval_palma(p)     = FairLoadDelivery.palma_ratio(p)

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
        "efficient" => eval_efficient(pshed),
        "palma"     => eval_palma(pshed),
    )

    @info "config $cfg: total_shed=$(round(sum(pshed), digits=3))  " *
          "palma=$(round(config_obj[cfg]["palma"], digits=3))"
end

#--------------------------------------------------------------------------------------------------------
# Save outputs
#--------------------------------------------------------------------------------------------------------

successful_configs = sort(collect(keys(config_obj)))

# 1) Wide summary: one row per config with totals + every fairness value.
summary_df = DataFrame(
    config_bits   = String[],
    total_pshed   = Float64[],
    obj_efficient = Float64[],
    obj_palma     = Float64[],
)
for cfg in successful_configs
    push!(summary_df, (cfg, sum(config_pshed[cfg]),
        config_obj[cfg]["efficient"], config_obj[cfg]["palma"]))
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
                title="Efficient: min Σ pshed", xrotation=90, legend=false)
plt_palma = bar(cfg_labels, summary_df.obj_palma;
                title="Palma: min",             xrotation=90, legend=false)
fig = plot(plt_eff, plt_palma;
           layout=(1, 2), size=(1400, 600),
           plot_title="Post-hoc fairness evaluation over $(length(successful_configs)) AC-feasible configs")
savefig(fig, joinpath(outdir, "brute_force_upper_level.svg"))
display(fig)

#--------------------------------------------------------------------------------------------------------
# Efficient vs Palma comparison on the de-duplicated representative configs
#--------------------------------------------------------------------------------------------------------

unique_summary = filter(row -> row.config_bits in unique_configs, summary_df)
sort!(unique_summary, :total_pshed)

# Drop configs where Palma is non-finite — they collapse to Inf when the bottom-40% pshed sum is 0.
finite_palma = filter(row -> isfinite(row.obj_palma), unique_summary)

if isempty(finite_palma)
    @warn "All distinct configs have Palma = Inf (bottom-40% pshed sum is zero everywhere)."
else
    best_eff_idx   = argmin(finite_palma.obj_efficient)
    best_palma_idx = argmin(finite_palma.obj_palma)

    scatter_eff_palma = scatter(finite_palma.obj_efficient, finite_palma.obj_palma;
        xlabel = "Σ pshed (lower → more efficient)",
        ylabel = "Palma ratio (lower → fairer top-vs-bottom)",
        title  = "Efficient vs Palma across $(nrow(finite_palma)) distinct operating points",
        markersize = 8, markeralpha = 0.7, color = :steelblue, label = "config",
        legend = :topright)
    for r in eachrow(finite_palma)
        annotate!(scatter_eff_palma, r.obj_efficient, r.obj_palma,
                  text(r.config_bits, 7, :black, :bottom))
    end
    scatter!(scatter_eff_palma,
        [finite_palma.obj_efficient[best_eff_idx]], [finite_palma.obj_palma[best_eff_idx]];
        markersize = 12, markershape = :star5, color = :red,
        label = "Efficient best (cfg $(finite_palma.config_bits[best_eff_idx]))")
    scatter!(scatter_eff_palma,
        [finite_palma.obj_efficient[best_palma_idx]], [finite_palma.obj_palma[best_palma_idx]];
        markersize = 12, markershape = :diamond, color = :green,
        label = "Palma best (cfg $(finite_palma.config_bits[best_palma_idx]))")
    savefig(scatter_eff_palma, joinpath(outdir, "efficient_vs_palma.svg"))
    display(scatter_eff_palma)

    CSV.write(joinpath(outdir, "efficient_vs_palma.csv"),
              select(finite_palma, :config_bits, :total_pshed, :obj_efficient, :obj_palma))

    @info "Efficient vs Palma best topologies:"
    @info "  Efficient best: cfg=$(finite_palma.config_bits[best_eff_idx]) " *
          "efficient=$(finite_palma.obj_efficient[best_eff_idx]) " *
          "palma=$(round(finite_palma.obj_palma[best_eff_idx], digits=4))"
    @info "  Palma best:     cfg=$(finite_palma.config_bits[best_palma_idx]) " *
          "efficient=$(finite_palma.obj_efficient[best_palma_idx]) " *
          "palma=$(round(finite_palma.obj_palma[best_palma_idx], digits=4))"
end
