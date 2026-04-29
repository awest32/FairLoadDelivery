#=============================================================================
 Multi-period α-sweep Pareto analysis

 Standalone script — does not depend on multi_period_trade_off_comparison.jl.
 For each fairness criterion × method, sweeps α ∈ ALPHA_VALUES:
   - α=0: pure efficiency (min weighted shed)
   - α=1: pure fairness
   - α in between: convex combination

 Solves at each α (single-level MIQCP + bilevel), evaluates ALL post-hoc
 metrics on the resulting solution, writes CSVs and plots to
 `results/<today>/multi_period_alpha_sweep/`.

 reg is forced to 0 inside this sweep so α is the only fairness/efficiency
 knob — avoids double-counting the efficiency term.
=============================================================================#

using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra, SparseArrays
using DataFrames
using CSV
using Dates
using Plots
using StatsPlots
using Statistics

const PMD = PowerModelsDistribution

#-----------------------------------------------------
# Network + multinetwork setup (mirrors the comparison script)
#-----------------------------------------------------
case = "ieee_13_aw_edit/motivation_c.dss"
dir = joinpath(@__DIR__, "../", "data")
file = joinpath(dir, case)

gen_cap = 0.8
source_pu = 1.03
switch_rating = 600.0
critical_loads = []

N_PERIODS = 3
LOAD_SCALE_FACTORS = [round(0.8 + 1.0 * exp(-((t - N_PERIODS/2)^2) / (2 * 4^2)), digits=3) for t in 0:N_PERIODS-1]
PEAK_TIME_COSTS = [round(8 + 22 * exp(-((t - N_PERIODS/2)^2) / (2 * 3^2)), digits=2) for t in 0:N_PERIODS-1]

eng, math = setup_network(file, gen_cap, source_pu, switch_rating, critical_loads)

function create_multinetwork_data(base_math::Dict{String,Any}, n_periods::Int, load_scales::Vector{Float64})
    @assert length(load_scales) == n_periods
    mn_data = Dict{String,Any}(
        "multinetwork" => true,
        "per_unit" => true,
        "data_model" => PMD.MATHEMATICAL,
        "nw" => Dict{String,Any}()
    )
    for key in ["baseMVA", "basekv", "bus_lookup", "settings"]
        if haskey(base_math, key)
            mn_data[key] = deepcopy(base_math[key])
        end
    end
    for t in 1:n_periods
        nw_id = string(t - 1)
        nw_data = deepcopy(base_math)
        delete!(nw_data, "multinetwork")
        scale = load_scales[t]
        for (load_id, load) in nw_data["load"]
            load["pd"] = load["pd"] .* scale
            load["qd"] = load["qd"] .* scale
        end
        nw_data["time_period"] = t
        nw_data["load_scale"] = scale
        mn_data["nw"][nw_id] = nw_data
    end
    return mn_data
end

mn_data = create_multinetwork_data(math, N_PERIODS, LOAD_SCALE_FACTORS)
nw_ids_sorted = sort(collect(keys(mn_data["nw"])), by=x->parse(Int, x))

# Initial weights (one per load; broadcast to T periods inside lower-level)
fair_weights_init = Float64[]
for (load_id, load) in math["load"]
    push!(fair_weights_init, load["weight"])
end

# Per-period upper bound on Σ_i weights_{t,i}. Each weight ∈ [1, 10]; budget
# of (n_loads - 1) + 10 lets one load spike while the rest stay near 1.
const WEIGHT_BUDGET = Float64(length(fair_weights_init) - 1 + 10)

# Output directory for this run
outdir = joinpath(@__DIR__, "..", "results", string(Dates.today()), "multi_period_alpha_sweep")
mkpath(outdir)

#-----------------------------------------------------
# Helpers
#-----------------------------------------------------

"""
Per-period per-load (shed, served, weights) from a PMD multinetwork solution.
"""
function per_period_shed_served_mn(soln, mn_data)
    T = length(nw_ids_sorted)
    shed = Vector{Vector{Float64}}(undef, T)
    served = Vector{Vector{Float64}}(undef, T)
    weights = Vector{Vector{Float64}}(undef, T)
    for (t, nw_id) in enumerate(nw_ids_sorted)
        load_ids = sort(parse.(Int, collect(keys(soln["solution"]["nw"][nw_id]["load"]))))
        shed_t = Float64[]; served_t = Float64[]; w_t = Float64[]
        for lid in load_ids
            pd_served = sum(soln["solution"]["nw"][nw_id]["load"][string(lid)]["pd"])
            pd_ref = sum(mn_data["nw"][nw_id]["load"][string(lid)]["pd"])
            w_val = get(mn_data["nw"][nw_id]["load"][string(lid)], "weight", 1.0)
            push!(shed_t, max(pd_ref - pd_served, 0.0))
            push!(served_t, pd_served)
            push!(w_t, Float64(w_val))
        end
        shed[t] = shed_t; served[t] = served_t; weights[t] = w_t
    end
    return shed, served, weights
end

"""
Solve the integer efficiency MILP with the current load weights in `mn_new`,
write resulting integer switch and block states back into `mn_new`. Used to
warm-start the next bilevel iteration's relaxed lower level.
"""
function warmstart_integer_topology!(mn_new; peak_time_costs=Float64[])
    t_start = time()
    milp_soln = solve_mn_mc_mld_switch_integer(mn_new, Gurobi.Optimizer;
                                                peak_time_costs=peak_time_costs)
    solve_time = time() - t_start
    status = get(milp_soln, "termination_status", MOI.OPTIMIZE_NOT_CALLED)
    if haskey(milp_soln, "solution") && haskey(milp_soln["solution"], "nw")
        for (nw_id, nw_soln) in milp_soln["solution"]["nw"]
            if !haskey(mn_new["nw"], nw_id); continue; end
            if haskey(nw_soln, "switch")
                for (sid, sw_data) in nw_soln["switch"]
                    if haskey(mn_new["nw"][nw_id]["switch"], sid) && haskey(sw_data, "state")
                        # Round MILP's near-integer float to exact Int —
                        # common.jl does `Int(switch["status"])` and throws
                        # InexactError on an unrounded float.
                        s_int = round(Int, sw_data["state"])
                        mn_new["nw"][nw_id]["switch"][sid]["state"] = s_int
                        mn_new["nw"][nw_id]["switch"][sid]["status"] = s_int
                        mn_new["nw"][nw_id]["switch"][sid]["dispatchable"] = 1
                    end
                end
            end
            if haskey(nw_soln, "block") && haskey(mn_new["nw"][nw_id], "block")
                for (bid, b_data) in nw_soln["block"]
                    if haskey(mn_new["nw"][nw_id]["block"], bid) && haskey(b_data, "status")
                        b_int = round(Int, b_data["status"])
                        mn_new["nw"][nw_id]["block"][bid]["state"]  = b_int
                        mn_new["nw"][nw_id]["block"][bid]["status"] = b_int
                    end
                end
            end
        end
    end
    return milp_soln, solve_time, status
end

"""
Run one bilevel iteration loop for a given criterion at a specific α.
Returns NamedTuple with final pshed, weights, status, iteration count, and
timing breakdown (lower/upper/milp).
"""
function run_bilevel_at_alpha(fair_func::String, mn_data_base, fair_weights_init, alpha::Float64;
                              iterations::Int=20, weight_budget::Float64=Inf)
    mn_new = deepcopy(mn_data_base)
    fair_weights = copy(fair_weights_init)
    nw_ids_local = sort(collect(keys(mn_new["nw"])), by=x->parse(Int, x))
    T = length(nw_ids_local)

    final_pshed = nothing
    final_wids = Int[]
    final_weights = Float64[]
    last_status = MOI.OPTIMIZE_NOT_CALLED
    completed = 0
    total_solve_time = 0.0
    lower_solve_time = 0.0
    upper_solve_time = 0.0
    milp_solve_time  = 0.0
    per_iteration_times = Float64[]

    for k in 1:iterations
        lower_t = @elapsed begin
            dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs =
                lower_level_soln_mn(mn_new, fair_weights, k)
        end
        lower_solve_time += lower_t
        n_l = length(weight_ids)

        # Reference demand from pristine mn_data_base so it stays fixed
        # across iterations (don't read from refs/mn_new — those can drift).
        pd_all_local = Float64[]
        for (nw, lid) in pshed_nw_ids
            push!(pd_all_local, sum(mn_data_base["nw"][string(nw)]["load"][string(lid)]["pd"]))
        end

        upper_t = @elapsed begin
            pshed_new, fair_w_vals, status = if fair_func == "proportional"
                proportional_fairness_load_shed(dpshed, pshed_val, weight_vals, pd_all_local;
                    critical_ids=Int[], weight_ids=weight_ids, peak_time_costs=PEAK_TIME_COSTS,
                    n_loads=n_l, reg=0.0, alpha=alpha, weight_budget=weight_budget)
            elseif fair_func == "min_max"
                min_max_load_shed(dpshed, pshed_val, weight_vals;
                    critical_ids=Int[], weight_ids=weight_ids, peak_time_costs=PEAK_TIME_COSTS,
                    n_loads=n_l, pd=pd_all_local, reg=0.0, alpha=alpha, weight_budget=weight_budget)
            elseif fair_func == "jain"
                jains_fairness_index(dpshed, pshed_val, weight_vals;
                    critical_ids=Int[], weight_ids=weight_ids, peak_time_costs=PEAK_TIME_COSTS,
                    n_loads=n_l, reg=0.0, pd=pd_all_local, alpha=alpha, weight_budget=weight_budget)
            elseif fair_func == "palma"
                lin_palma_reformulated(dpshed, pshed_val, weight_vals, pd_all_local;
                    critical_ids=Int[], weight_ids=weight_ids, peak_time_costs=PEAK_TIME_COSTS,
                    n_loads=n_l, reg=0.0, alpha=alpha, weight_budget=weight_budget)
            else
                error("Unknown fair_func: $fair_func")
            end
        end
        upper_solve_time += upper_t

        last_status = status
        if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED,
                     MOI.ALMOST_OPTIMAL, MOI.TIME_LIMIT]
            @warn "[$fair_func α=$alpha] iter $k stopped with status=$status"
            total_solve_time += lower_t + upper_t
            push!(per_iteration_times, lower_t + upper_t)
            break
        end
        completed = k

        for (t, nw_id) in enumerate(nw_ids_local)
            offset = (t - 1) * n_l
            for (j, lid) in enumerate(weight_ids)
                mn_new["nw"][nw_id]["load"][string(lid)]["weight"] = fair_w_vals[offset + j]
            end
        end

        # Integer MILP warm-start for next iteration's lower level
        _, milp_t, _ = warmstart_integer_topology!(mn_new;
            peak_time_costs=PEAK_TIME_COSTS)
        milp_solve_time += milp_t
        total_solve_time += lower_t + upper_t + milp_t
        push!(per_iteration_times, lower_t + upper_t + milp_t)

        fair_weights = copy(fair_w_vals)
        final_pshed = copy(pshed_val)
        final_wids = weight_ids
        final_weights = copy(fair_w_vals)
    end
    return (pshed = final_pshed, weight_ids = final_wids,
            final_weights = final_weights, status = last_status,
            iterations = completed, total_solve_time = total_solve_time,
            lower_solve_time = lower_solve_time, upper_solve_time = upper_solve_time,
            milp_solve_time = milp_solve_time,
            per_iteration_times = per_iteration_times)
end

"""
Evaluate all 5 post-hoc metrics on per-period (shed, served, weights).
"""
function eval_all_metrics(shed, served, weights=nothing)
    return (
        weighted_shed = weighted_shed_mn(shed; peak_time_costs=PEAK_TIME_COSTS, weights=weights),
        jain          = jain_mn(served;        peak_time_costs=PEAK_TIME_COSTS, weights=weights),
        min_max       = min_max_mn(shed;       peak_time_costs=PEAK_TIME_COSTS, weights=weights),
        proportional  = proportional_mn(served; peak_time_costs=PEAK_TIME_COSTS, weights=weights),
        palma         = palma_mn(shed;         peak_time_costs=PEAK_TIME_COSTS, weights=weights),
    )
end

"""
Build per-period per-load (shed, served, weights) from a flat pshed vector
(period-major: offset = (t-1)*n_loads).
"""
function shed_served_from_pshed_vec(pshed::Vector{Float64}, weight_ids::Vector{Int},
                                     final_weights::Vector{Float64})
    n_l = length(weight_ids)
    T = length(nw_ids_sorted)
    shed = Vector{Vector{Float64}}(undef, T)
    served = Vector{Vector{Float64}}(undef, T)
    weights = Vector{Vector{Float64}}(undef, T)
    for (t, nw_id) in enumerate(nw_ids_sorted)
        offset = (t - 1) * n_l
        shed_t = Float64[]; served_t = Float64[]; w_t = Float64[]
        for (j, lid) in enumerate(weight_ids)
            pd_ref = sum(mn_data["nw"][nw_id]["load"][string(lid)]["pd"])
            s = max(pshed[offset + j], 0.0)
            push!(shed_t, s); push!(served_t, max(pd_ref - s, 0.0))
            push!(w_t, Float64(final_weights[offset + j]))
        end
        shed[t] = shed_t; served[t] = served_t; weights[t] = w_t
    end
    return shed, served, weights
end

#-----------------------------------------------------
# α-sweep
#-----------------------------------------------------
const ALPHA_VALUES = [0.0, 0.5, 1.0]

pareto_df = DataFrame(
    criterion = String[], method = String[], alpha = Float64[],
    weighted_shed = Float64[], jain = Float64[], min_max = Float64[],
    proportional = Float64[], palma = Float64[], status = String[],
    total_solve_time = Float64[],
    lower_solve_time = Float64[],
    upper_solve_time = Float64[],
    iterations = Int[],
)

const SINGLE_LEVEL_SOLVERS = Dict(
    "jain"         => solve_mn_mc_mld_jain_integer,
    "min_max"      => solve_mn_mc_mld_min_max_integer,
    "proportional" => solve_mn_mc_mld_proportional_fairness_integer,
    "palma"        => solve_mn_mc_mld_palma_integer,
)

for criterion in ["jain", "min_max", "proportional", "palma"]
    solver_fn = SINGLE_LEVEL_SOLVERS[criterion]
    for α in ALPHA_VALUES
        @info "=== Pareto sweep: criterion=$criterion, α=$α ==="

        # Single-level MIQCP with this α, reg=0
        try
            sl_time = @elapsed sl_soln = solver_fn(mn_data, Gurobi.Optimizer;
                peak_time_costs=PEAK_TIME_COSTS, reg=0.0, alpha=α)
            sl_shed, sl_served, sl_w = per_period_shed_served_mn(sl_soln, mn_data)
            m = eval_all_metrics(sl_shed, sl_served, sl_w)
            push!(pareto_df, (criterion, "single_level", α,
                m.weighted_shed, m.jain, m.min_max, m.proportional, m.palma,
                string(sl_soln["termination_status"]),
                sl_time, 0.0, 0.0, 1))
            @info "    single-level: $(round(sl_time, digits=2))s"
        catch e
            @warn "[$criterion α=$α single-level] failed: $e"
            push!(pareto_df, (criterion, "single_level", α,
                NaN, NaN, NaN, NaN, NaN, "ERROR",
                NaN, 0.0, 0.0, 0))
        end

        # Bilevel with this α, reg=0, per-period weight budget
        try
            bl = run_bilevel_at_alpha(criterion, mn_data, fair_weights_init, α;
                                      weight_budget=WEIGHT_BUDGET)
            if isnothing(bl.pshed)
                push!(pareto_df, (criterion, "bilevel", α,
                    NaN, NaN, NaN, NaN, NaN, string(bl.status),
                    bl.total_solve_time, bl.lower_solve_time, bl.upper_solve_time,
                    bl.iterations))
            else
                bl_shed, bl_served, bl_w = shed_served_from_pshed_vec(bl.pshed, bl.weight_ids, bl.final_weights)
                m = eval_all_metrics(bl_shed, bl_served, bl_w)
                push!(pareto_df, (criterion, "bilevel", α,
                    m.weighted_shed, m.jain, m.min_max, m.proportional, m.palma,
                    string(bl.status),
                    bl.total_solve_time, bl.lower_solve_time, bl.upper_solve_time,
                    bl.iterations))
            end
            @info "    bilevel: $(round(bl.total_solve_time, digits=2))s total ($(bl.iterations) iters; lower=$(round(bl.lower_solve_time, digits=2))s, upper=$(round(bl.upper_solve_time, digits=2))s)"
        catch e
            @warn "[$criterion α=$α bilevel] failed: $e"
            push!(pareto_df, (criterion, "bilevel", α,
                NaN, NaN, NaN, NaN, NaN, "ERROR",
                NaN, NaN, NaN, 0))
        end
    end
end

CSV.write(joinpath(outdir, "pareto_sweep.csv"), pareto_df)

#-----------------------------------------------------
# Plots
#-----------------------------------------------------

# Primary Pareto plot: one subplot per criterion, own metric vs shed.
const METRIC_META = Dict(
    "jain"         => (col = :jain,         label = "λ-weighted Jain ∈ [1/n, 1] (↑)", better = :up),
    "min_max"      => (col = :min_max,      label = "Σ λ·max shed (↓)",               better = :down),
    "proportional" => (col = :proportional, label = "Σ λ·Σ log(served) (↑)",          better = :up),
    "palma"        => (col = :palma,        label = "Σ λ·Palma (↓)",                  better = :down),
)

pareto_plots = []
for criterion in ["jain", "min_max", "proportional", "palma"]
    meta = METRIC_META[criterion]
    sub = pareto_df[pareto_df.criterion .== criterion, :]
    sl  = sub[sub.method .== "single_level", :]
    bl  = sub[sub.method .== "bilevel", :]
    mk = FAIR_FUNC_MARKERS[criterion]
    cl = FAIR_FUNC_COLORS[criterion]

    p = plot(title = "$(FAIR_FUNC_LABELS[criterion])" *
             (meta.better == :up ? " (↑ fairer)" : " (↓ fairer)"),
             xlabel = "Σ λ·Σ pshed", ylabel = meta.label)
    plot!(p, sl.weighted_shed, sl[!, meta.col]; label = "single-level",
          marker=mk, color=cl, linewidth=2, linestyle=:solid, markersize=7)
    plot!(p, bl.weighted_shed, bl[!, meta.col]; label = "bilevel",
          marker=mk, markercolor=:white, markerstrokecolor=cl,
          color=cl, linewidth=2, linestyle=:dash, markersize=7)
    for r in eachrow(sl)
        annotate!(p, r.weighted_shed, r[meta.col], text("α=$(r.alpha)", :bottom, 7, cl))
    end
    for r in eachrow(bl)
        annotate!(p, r.weighted_shed, r[meta.col], text("α=$(r.alpha)", :top, 7, cl))
    end
    push!(pareto_plots, p)
end
pareto_fig = plot(pareto_plots...; layout=(2, 2), size=(1100, 900),
    plot_title = "Pareto front: α-sweep over fairness/efficiency trade-off")
savefig(pareto_fig, joinpath(outdir, "pareto_fronts.svg"))
display(pareto_fig)

# Solve-time plot: log-scale y-axis, single-level vs bilevel across α.
time_plots = []
for criterion in ["jain", "min_max", "proportional", "palma"]
    sub = pareto_df[pareto_df.criterion .== criterion, :]
    sl  = sub[sub.method .== "single_level", :]
    bl  = sub[sub.method .== "bilevel", :]
    cl  = FAIR_FUNC_COLORS[criterion]
    p = plot(title = "$(FAIR_FUNC_LABELS[criterion]) — solve time",
             xlabel = "α", ylabel = "Total solve time (s)",
             yscale = :log10, legend = :topright)
    plot!(p, sl.alpha, max.(sl.total_solve_time, 1e-3);
          label = "single-level MIQCP", marker = :square,
          color = cl, linewidth = 2, linestyle = :solid, markersize = 7)
    plot!(p, bl.alpha, max.(bl.total_solve_time, 1e-3);
          label = "bilevel (Σ iters)", marker = :circle,
          markercolor = :white, markerstrokecolor = cl,
          color = cl, linewidth = 2, linestyle = :dash, markersize = 7)
    for r in eachrow(bl)
        annotate!(p, r.alpha, max(r.total_solve_time, 1e-3),
                  text("$(r.iterations) it", :top, 7, cl))
    end
    push!(time_plots, p)
end
time_fig = plot(time_plots...; layout=(2, 2), size=(1100, 900),
    plot_title = "Solve-time comparison (log scale): single-level vs bilevel")
savefig(time_fig, joinpath(outdir, "solve_times.svg"))
display(time_fig)

# Cross-metric matrix: row = criterion optimized, col = metric evaluated.
const METRIC_COLS = [:weighted_shed, :jain, :min_max, :proportional, :palma]
const METRIC_LABELS = ["Σ λ·Σ pshed", "λ-avg Jain ∈ [1/n,1]", "Σ λ·max shed", "Σ λ·Σ log(served)", "Σ λ·Palma"]
cross_plots = []
for criterion in ["jain", "min_max", "proportional", "palma"]
    sub = pareto_df[pareto_df.criterion .== criterion, :]
    sl  = sub[sub.method .== "single_level", :]
    bl  = sub[sub.method .== "bilevel", :]
    mk = FAIR_FUNC_MARKERS[criterion]
    cl = FAIR_FUNC_COLORS[criterion]
    for (col, lbl) in zip(METRIC_COLS, METRIC_LABELS)
        p = plot(title = "optimized: $(FAIR_FUNC_LABELS[criterion]) → $lbl",
                 xlabel = "α", ylabel = lbl, legend = false, titlefontsize=8)
        plot!(p, sl.alpha, sl[!, col]; marker=mk, color=cl, linewidth=2, linestyle=:solid)
        plot!(p, bl.alpha, bl[!, col]; marker=mk, color=cl, linewidth=2, linestyle=:dash,
              markerstrokecolor=cl, markercolor=:white)
        push!(cross_plots, p)
    end
end
cross_fig = plot(cross_plots...; layout=(4, 5), size=(1600, 1200),
    plot_title = "Cross-metric evaluation (rows=criterion optimized, cols=metric evaluated)")
savefig(cross_fig, joinpath(outdir, "cross_metric_matrix.svg"))
display(cross_fig)
