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

# Load the test case
case = "ieee_13_aw_edit/motivation_c.dss"
dir = joinpath(@__DIR__, "../", "data")
file = joinpath(dir, case)

# Test parameters
gen_cap = 0.8
source_pu = 1.03
switch_rating = 600.0
critical_loads = []

# Multiperiod parameters
N_PERIODS = 3
# Gaussian-ish load profile peaking around the middle period
LOAD_SCALE_FACTORS = [round(0.8 + 1.0 * exp(-((t - N_PERIODS/2)^2) / (2 * 4^2)), digits=3) for t in 0:N_PERIODS-1]
# Peak-time energy cost per period, used to weight each period's fairness term
PEAK_TIME_COSTS = [round(8 + 22 * exp(-((t - N_PERIODS/2)^2) / (2 * 3^2)), digits=2) for t in 0:N_PERIODS-1]

# Load the data
eng, math = setup_network(file, gen_cap, source_pu, switch_rating, critical_loads)

"""
Replicate a single-period math dict into a multinetwork dict with per-period
load scaling. Mirrors script/multiperiod/compare_fair_funcs_multiperiod.jl.
"""
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

total_demand_all_periods = sum(
    sum(sum(load["pd"]) for (_, load) in mn_data["nw"][nw_id]["load"])
    for nw_id in nw_ids_sorted
)

# Output directory for figures: results/<today>/multi_period_trade_off_comparison
outdir = joinpath(@__DIR__, "..", "results", string(Dates.today()), "multi_period_trade_off_comparison")
mkpath(outdir)

#-----------------------------------------------------
# Single-level multi-period solves for each fairness function
#------------------------------------------------------
jain_soln = solve_mn_mc_mld_jain_integer(mn_data, Gurobi.Optimizer; peak_time_costs=PEAK_TIME_COSTS)
min_max_soln = solve_mn_mc_mld_min_max_integer(mn_data, Gurobi.Optimizer; peak_time_costs=PEAK_TIME_COSTS)
proportional_soln = solve_mn_mc_mld_proportional_fairness_integer(mn_data, Gurobi.Optimizer; peak_time_costs=PEAK_TIME_COSTS)
efficient_soln = solve_mn_mc_mld_switch_integer(mn_data, Gurobi.Optimizer; peak_time_costs=PEAK_TIME_COSTS)
palma_soln = solve_mn_mc_mld_palma_integer(mn_data, Gurobi.Optimizer; peak_time_costs=PEAK_TIME_COSTS)

function total_served_mn(soln)
    s = 0.0
    for (_, nw_soln) in soln["solution"]["nw"]
        if haskey(nw_soln, "load")
            for (_, load) in nw_soln["load"]
                s += sum(load["pd"])
            end
        end
    end
    return s
end

function per_period_served_mn(soln)
    per_period = Float64[]
    for nw_id in nw_ids_sorted
        nw_soln = soln["solution"]["nw"][nw_id]
        s = 0.0
        if haskey(nw_soln, "load")
            for (_, load) in nw_soln["load"]
                s += sum(load["pd"])
            end
        end
        push!(per_period, s)
    end
    return per_period
end

"""
Per-period per-load (shed, served) extracted from a PMD multinetwork solution.
Each returned entry is a length-T vector of per-load vectors. Shed is
pd_ref - pd_served; served is pd_served directly from the solution.
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
Bilevel version: reconstruct per-period per-load (shed, served) from the
final lower-level pshed vector (period-major layout: offset = (t-1)*n_loads).
Returns (nothing, nothing) if no bilevel iterations completed.
"""
function per_period_shed_served_bilevel(res, mn_data)
    isempty(res["pshed_lower_history"]) && return nothing, nothing, nothing
    pshed = res["pshed_lower_history"][end]
    weight_ids = res["final_weight_ids"]
    final_w = res["final_weights"]
    n_loads = length(weight_ids)
    T = length(nw_ids_sorted)
    shed = Vector{Vector{Float64}}(undef, T)
    served = Vector{Vector{Float64}}(undef, T)
    weights = Vector{Vector{Float64}}(undef, T)
    for (t, nw_id) in enumerate(nw_ids_sorted)
        offset = (t - 1) * n_loads
        shed_t = Float64[]; served_t = Float64[]; w_t = Float64[]
        for (j, lid) in enumerate(weight_ids)
            pd_ref = sum(mn_data["nw"][nw_id]["load"][string(lid)]["pd"])
            s = max(pshed[offset + j], 0.0)
            push!(shed_t, s)
            push!(served_t, max(pd_ref - s, 0.0))
            push!(w_t, Float64(final_w[offset + j]))
        end
        shed[t] = shed_t; served[t] = served_t; weights[t] = w_t
    end
    return shed, served, weights
end

jain_pserved = total_served_mn(jain_soln)
min_max_pserved = total_served_mn(min_max_soln)
proportional_pserved = total_served_mn(proportional_soln)
efficient_pserved = total_served_mn(efficient_soln)
palma_pserved = total_served_mn(palma_soln)

pserved_df = DataFrame(
    fairness_function = ["jain", "min_max", "proportional", "efficient", "palma"],
    load_served_total = [jain_pserved, min_max_pserved, proportional_pserved, efficient_pserved, palma_pserved],
    load_shed_total = total_demand_all_periods .- [jain_pserved, min_max_pserved, proportional_pserved, efficient_pserved, palma_pserved],
)
total_shed_plot = @df pserved_df bar(:fairness_function, :load_shed_total,
    title="Total Load Shed Across $N_PERIODS Periods",
    legend=false)
xlabel!("Fairness Function")
ylabel!("Total Load Shed (p.u., summed over periods)")
savefig(total_shed_plot, joinpath(outdir, "single_level_total_load_shed.svg"))
display(total_shed_plot)

# Stacked per-period served to see the temporal distribution
per_period_matrix = hcat(
    per_period_served_mn(jain_soln),
    per_period_served_mn(min_max_soln),
    per_period_served_mn(proportional_soln),
    per_period_served_mn(efficient_soln),
    per_period_served_mn(palma_soln),
)
per_period_plot = groupedbar(per_period_matrix,
    bar_position = :dodge,
    bar_width = 0.7,
    xticks = (1:N_PERIODS, ["t=$(nw_ids_sorted[i])" for i in 1:N_PERIODS]),
    label = ["jain" "min_max" "proportional" "efficient" "palma"],
    xlabel = "Period",
    ylabel = "Load Served (p.u.)",
    title = "Per-Period Load Served by Fairness Function")
savefig(per_period_plot, joinpath(outdir, "single_level_per_period_load_served.svg"))
display(per_period_plot)

#-----------------------------------------------------
# Bilevel results with multiperiod lower level
#------------------------------------------------------
# Get initial weights (one per load; broadcast to T periods inside lower-level)
fair_weights_init = Float64[]
for (load_id, load) in math["load"]
    push!(fair_weights_init, load["weight"])
end

# Per-period upper bound on Σ_i weights_{t,i}. Each weight ∈ [1, 10]; a budget
# of (n_loads - 1) + 10 lets one load spike to ~10 while the rest stay near 1.
# Floor (all at 1) = n_loads, so this keeps the problem feasible. This is
# applied in BOTH the main bilevel loop and the later α-sweep.
const WEIGHT_BUDGET = Float64(length(fair_weights_init) - 1 + 10)

iterations = 20
bilevel_results = Dict{String, Any}()

"""
Solve the integer efficiency MILP (`solve_mn_mc_mld_switch_integer`) using the
current load weights in `mn_new`, then write the resulting integer switch
states and block states back into `mn_new` so the next iteration's lower-level
relaxation sees an integer-feasible topology as its starting point.

This is called AFTER the upper-level fairness solve writes new weights into
`mn_new`. The integer MILP minimizes weighted shed given those weights, so the
topology it returns is the efficient switch configuration for the current
weight vector.

Mutates `mn_new` in place. Returns (milp_soln, solve_time_seconds, status).
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
                        # Round MILP's near-integer float (e.g. 0.99999978) to
                        # an exact Int — common.jl does `Int(switch["status"])`
                        # and will throw InexactError on an unrounded float.
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

for fair_func in ["proportional", "efficiency", "min_max", "jain", "palma"]
    mn_new = deepcopy(mn_data)
    fair_weights = copy(fair_weights_init)
    nw_ids = sort(collect(keys(mn_new["nw"])), by=x->parse(Int, x))
    T = length(nw_ids)

    pshed_lower_level = Float64[]
    pshed_upper_level = Float64[]
    final_weight_ids = Int[]
    final_weights = Float64[]
    completed_iterations = 0
    last_status = MOI.OPTIMIZE_NOT_CALLED
    prev_weights = repeat(fair_weights, T)
    prev_pshed = Float64[]
    max_delta_weights = NaN
    max_delta_pshed = NaN
    iteration_log = []
    # Per-iteration histories: each entry is a length T*N vector, ordered period-major
    weights_history = Vector{Vector{Float64}}()
    pshed_lower_history = Vector{Vector{Float64}}()
    pshed_upper_history = Vector{Vector{Float64}}()

    for k in 1:iterations
        # Solve multiperiod lower-level and get (T*N) x (T*N) Jacobian
        dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs = lower_level_soln_mn(mn_new, fair_weights, k)

        n_loads = length(weight_ids)

        # Per-period pd reference values matching pshed ordering. Read from
        # `mn_data` (pristine outer-scope multinetwork dict) — NOT from `refs`
        # or `mn_new`, which can pick up mutations across iterations (e.g. the
        # MILP warm-start updates switch/block states, and some PMD internals
        # may rewrite load pd). The reference demand should be the *maximum*
        # load the bus needs, fixed across iterations.
        pd_all = Float64[]
        for (nw, lid) in pshed_nw_ids
            push!(pd_all, sum(mn_data["nw"][string(nw)]["load"][string(lid)]["pd"]))
        end

        # Apply fairness function on the full (T*N) pshed vector.
        # Note: signatures differ — jain and min_max take critical_ids/weight_ids
        # positionally; proportional, efficient, and palma take them as kwargs.
        if fair_func == "proportional"
            pshed_new, fair_weight_vals, status = proportional_fairness_load_shed(dpshed, pshed_val, weight_vals, pd_all; critical_ids=Int[], weight_ids=weight_ids, peak_time_costs=PEAK_TIME_COSTS, n_loads=n_loads, weight_budget=WEIGHT_BUDGET)
        elseif fair_func == "efficiency"
            pshed_new, fair_weight_vals, status = efficient_load_shed(dpshed, pshed_val, weight_vals; critical_ids=Int[], weight_ids=weight_ids, peak_time_costs=PEAK_TIME_COSTS, n_loads=n_loads)
        elseif fair_func == "min_max"
            pshed_new, fair_weight_vals, status = min_max_load_shed(dpshed, pshed_val, weight_vals; critical_ids=Int[], weight_ids=weight_ids, peak_time_costs=PEAK_TIME_COSTS, n_loads=n_loads, pd=pd_all, weight_budget=WEIGHT_BUDGET)
        elseif fair_func == "jain"
            pshed_new, fair_weight_vals, status = jains_fairness_index(dpshed, pshed_val, weight_vals; critical_ids=Int[], weight_ids=weight_ids, peak_time_costs=PEAK_TIME_COSTS, n_loads=n_loads, pd=pd_all, weight_budget=WEIGHT_BUDGET)
        elseif fair_func == "palma"
            pshed_new, fair_weight_vals, status = lin_palma_reformulated(dpshed, pshed_val, weight_vals, pd_all; critical_ids=Int[], weight_ids=weight_ids, peak_time_costs=PEAK_TIME_COSTS, n_loads=n_loads, weight_budget=WEIGHT_BUDGET)
        else
            error("Unknown fairness function: $fair_func")
        end

        last_status = status
        @info "[$fair_func] Iteration $k: upper-level status = $status"
        if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.TIME_LIMIT]
            @warn "[$fair_func] Iteration $k/$iterations: upper-level infeasible (status=$status) — stopping"
            break
        end
        completed_iterations = k

        if k != 1
            max_delta_weights = maximum(abs.(fair_weight_vals .- prev_weights))
        end
        if !isempty(prev_pshed)
            max_delta_pshed = maximum(abs.(pshed_new .- prev_pshed))
        end
        dw = fair_weight_vals .- prev_weights
        prev_weights = copy(fair_weight_vals)
        prev_pshed = copy(pshed_new)

        # Push per-period weights back into mn_new
        for (t, nw_id) in enumerate(nw_ids)
            nw_data = mn_new["nw"][nw_id]
            offset = (t - 1) * n_loads
            for (j, lid) in enumerate(weight_ids)
                nw_data["load"][string(lid)]["weight"] = fair_weight_vals[offset + j]
            end
        end

        # Integer MILP warm-start: solve efficiency MILP with the new weights,
        # write integer switch/block states into mn_new. The next iteration's
        # lower-level relaxation will start from this topology.
        milp_soln, milp_time, milp_status = warmstart_integer_topology!(mn_new;
            peak_time_costs=PEAK_TIME_COSTS)
        @info "[$fair_func] iter $k: integer MILP warm-start $(round(milp_time, digits=2))s, status=$milp_status"

        fair_weights = copy(fair_weight_vals)
        final_weight_ids = weight_ids
        final_weights = fair_weight_vals

        push!(pshed_lower_level, sum(pshed_val))
        push!(pshed_upper_level, sum(pshed_new))

        # Keep full per-iteration vectors for plotting the trajectory
        push!(weights_history, copy(fair_weight_vals))
        push!(pshed_lower_history, copy(pshed_val))
        push!(pshed_upper_history, copy(pshed_new))

        push!(iteration_log, (
            iteration = k,
            status = string(status),
            pshed_lower = sum(pshed_val),
            pshed_upper = sum(pshed_new),
            max_delta_w = maximum(abs.(dw)),
            mean_delta_w = mean(abs.(dw)),
            w_min = minimum(fair_weight_vals),
            w_max = maximum(fair_weight_vals),
            jac_max = maximum(abs.(dpshed)),
            jac_nnz = count(!=(0.0), dpshed),
            n_pshed_zero = count(x -> x < 1e-6, pshed_val),
            n_pshed_total = length(pshed_val),
            milp_warmstart_time = milp_time,
            milp_warmstart_status = string(milp_status),
        ))
    end

    bilevel_results[fair_func] = Dict(
        "pshed_lower_level" => pshed_lower_level,
        "pshed_upper_level" => pshed_upper_level,
        "final_weight_ids"  => final_weight_ids,
        "final_weights"     => final_weights,
        "completed_iterations" => completed_iterations,
        "last_status"       => last_status,
        "max_delta_weights" => max_delta_weights,
        "max_delta_pshed"   => max_delta_pshed,
        "iteration_log"     => iteration_log,
        "weights_history"      => weights_history,
        "pshed_lower_history"  => pshed_lower_history,
        "pshed_upper_history"  => pshed_upper_history,
    )
end

#-----------------------------------------------------
# Per-iteration evolution plots (weights + load shed)
#------------------------------------------------------
"""
Build an (K x N) matrix where K is the number of iterations and each column is
a (load, period) series. Returns (matrix, column_labels).
Series is split across periods so the plot has one subplot per period.
"""
function _series_by_period(history::Vector{Vector{Float64}}, weight_ids::Vector{Int}, T::Int)
    n_loads = length(weight_ids)
    K = length(history)
    mat = zeros(K, T * n_loads)
    for k in 1:K
        mat[k, :] .= history[k]
    end
    return mat, n_loads
end

for fair_func in ["proportional", "efficiency", "min_max", "jain", "palma"]
    res = bilevel_results[fair_func]
    if isempty(res["weights_history"])
        @warn "[$fair_func] no iterations completed — skipping evolution plots"
        continue
    end
    weight_ids = res["final_weight_ids"]
    K = length(res["weights_history"])
    T = length(nw_ids_sorted)
    n_loads = length(weight_ids)

    w_mat, _ = _series_by_period(res["weights_history"], weight_ids, T)
    p_mat, _ = _series_by_period(res["pshed_lower_history"], weight_ids, T)

    # One subplot per period for weights
    weight_plots = []
    for t in 1:T
        offset = (t - 1) * n_loads
        cols = offset .+ (1:n_loads)
        series = w_mat[:, cols]
        labels = reshape(["L$(weight_ids[j])" for j in 1:n_loads], 1, n_loads)
        p = plot(1:K, series,
            xlabel = "Iteration", ylabel = "Weight",
            title = "$fair_func — period $(nw_ids_sorted[t])",
            label = labels, legend = t == T ? :outerright : false,
            marker = :circle, markersize = 3)
        push!(weight_plots, p)
    end
    weights_fig = plot(weight_plots...; layout=(T, 1), size=(800, 220*T),
                  plot_title = "Weights over iterations ($fair_func)")
    savefig(weights_fig, joinpath(outdir, "weights_over_iterations_$(fair_func).svg"))
    display(weights_fig)

    # One subplot per period for load shed (lower-level pshed)
    pshed_plots = []
    for t in 1:T
        offset = (t - 1) * n_loads
        cols = offset .+ (1:n_loads)
        series = p_mat[:, cols]
        labels = reshape(["L$(weight_ids[j])" for j in 1:n_loads], 1, n_loads)
        p = plot(1:K, series,
            xlabel = "Iteration", ylabel = "pshed (p.u.)",
            title = "$fair_func — period $(nw_ids_sorted[t])",
            label = labels, legend = t == T ? :outerright : false,
            marker = :circle, markersize = 3)
        push!(pshed_plots, p)
    end
    pshed_fig = plot(pshed_plots...; layout=(T, 1), size=(800, 220*T),
                  plot_title = "Lower-level load shed over iterations ($fair_func)")
    savefig(pshed_fig, joinpath(outdir, "lower_level_pshed_over_iterations_$(fair_func).svg"))
    display(pshed_fig)
end

# Aggregate totals-per-iteration across all fairness functions for a quick overview
agg_lower = plot(title = "Total lower-level pshed per iteration",
                 xlabel = "Iteration", ylabel = "Σ pshed")
agg_upper = plot(title = "Total upper-level pshed per iteration",
                 xlabel = "Iteration", ylabel = "Σ pshed")
for fair_func in ["proportional", "efficiency", "min_max", "jain", "palma"]
    res = bilevel_results[fair_func]
    if !isempty(res["pshed_lower_level"])
        plot!(agg_lower, 1:length(res["pshed_lower_level"]), res["pshed_lower_level"];
              label = FAIR_FUNC_LABELS[fair_func], marker = FAIR_FUNC_MARKERS[fair_func],
              color = FAIR_FUNC_COLORS[fair_func], linewidth = 2)
    end
    if !isempty(res["pshed_upper_level"])
        plot!(agg_upper, 1:length(res["pshed_upper_level"]), res["pshed_upper_level"];
              label = FAIR_FUNC_LABELS[fair_func], marker = FAIR_FUNC_MARKERS[fair_func],
              color = FAIR_FUNC_COLORS[fair_func], linewidth = 2)
    end
end
agg_fig = plot(agg_lower, agg_upper; layout=(2, 1), size=(800, 600))
savefig(agg_fig, joinpath(outdir, "aggregate_pshed_per_iteration.svg"))
display(agg_fig)

# Compare single-level total shed with bilevel lower/upper totals.
compare_df = DataFrame(
    fairness_function = String[],
    single_level_shed = Float64[],
    bilevel_lower_shed = Float64[],
    bilevel_upper_shed = Float64[],
)
single_level_shed = Dict(
    "proportional" => total_demand_all_periods - proportional_pserved,
    "efficiency"   => total_demand_all_periods - efficient_pserved,
    "min_max"      => total_demand_all_periods - min_max_pserved,
    "jain"         => total_demand_all_periods - jain_pserved,
    "palma"        => total_demand_all_periods - palma_pserved,
)
for f in ["proportional", "efficiency", "min_max", "jain", "palma"]
    bl_lower = isempty(bilevel_results[f]["pshed_lower_level"]) ? NaN : bilevel_results[f]["pshed_lower_level"][end]
    bl_upper = isempty(bilevel_results[f]["pshed_upper_level"]) ? NaN : bilevel_results[f]["pshed_upper_level"][end]
    push!(compare_df, (f, single_level_shed[f], bl_lower, bl_upper))
end

compare_plot = @df compare_df groupedbar(
    :fairness_function,
    hcat(:single_level_shed, :bilevel_lower_shed, :bilevel_upper_shed),
    labels = ["Single-level MILP" "Bilevel lower" "Bilevel upper"],
    bar_position = :dodge,
    title = "Single-level vs Bilevel Total Load Shed ($N_PERIODS periods)",
    xlabel = "Fairness Function",
    ylabel = "Total Load Shed (p.u.)",
)
savefig(compare_plot, joinpath(outdir, "single_vs_bilevel_total_load_shed.svg"))
display(compare_plot)

#-----------------------------------------------------
# Shed-vs-fairness comparison (three discrete solutions, NOT a Pareto front)
#
# Research question: does decoupling fairness (upper) from efficient
# operation (lower) produce less load shed AND more fair outcomes than
# either endpoint (efficiency MILP / single-level fairness MILP)?
#
# One subplot per fairness criterion. Three points per subplot:
#   - efficiency MILP (min Σ pshed)
#   - fairness MILP for that criterion
#   - bilevel with that criterion as upper-level objective
#
# An actual Pareto front would require a parameter sweep (e.g. varying the
# regularizer `reg` from 0 toward large, or an α-weighted convex combination
# of fairness and efficiency objectives) producing a curve of non-dominated
# points. This is 3 points, not a curve — it tells you which corner of the
# shed/fairness space each method lands in.
#------------------------------------------------------

# Metric metadata: which module function to call and which input it needs
const COMPARISON_METRICS = Dict(
    "jain"         => (fn = jain_mn,         use = :served, label = "Σ λ·Jain (↑)",           better = :up),
    "min_max"      => (fn = min_max_mn,      use = :shed,   label = "Σ λ·max shed (↓)",        better = :down),
    "proportional" => (fn = proportional_mn, use = :served, label = "Σ λ·Σ log(served) (↑)",   better = :up),
    "palma"        => (fn = palma_mn,        use = :shed,   label = "Σ λ·Palma ratio (↓)",     better = :down),
)

single_level_solns = Dict(
    "jain" => jain_soln, "min_max" => min_max_soln,
    "proportional" => proportional_soln, "palma" => palma_soln,
)

eff_shed, eff_served, eff_weights = per_period_shed_served_mn(efficient_soln, mn_data)

comparison_df = DataFrame(
    criterion = String[], approach = String[],
    weighted_shed = Float64[], fairness_metric = Float64[],
)

comparison_plots = []
for criterion in ["jain", "min_max", "proportional", "palma"]
    meta = COMPARISON_METRICS[criterion]

    # Efficiency MILP endpoint
    eff_x = weighted_shed_mn(eff_shed; peak_time_costs=PEAK_TIME_COSTS, weights=eff_weights)
    eff_y = meta.fn(meta.use == :served ? eff_served : eff_shed;
                    peak_time_costs=PEAK_TIME_COSTS, weights=eff_weights)

    # Fairness MILP endpoint for this criterion
    fair_shed, fair_served, fair_w = per_period_shed_served_mn(single_level_solns[criterion], mn_data)
    fair_x = weighted_shed_mn(fair_shed; peak_time_costs=PEAK_TIME_COSTS, weights=fair_w)
    fair_y = meta.fn(meta.use == :served ? fair_served : fair_shed;
                     peak_time_costs=PEAK_TIME_COSTS, weights=fair_w)

    # Bilevel point
    bl_shed, bl_served, bl_w = per_period_shed_served_bilevel(bilevel_results[criterion], mn_data)
    bl_x = isnothing(bl_shed) ? NaN : weighted_shed_mn(bl_shed; peak_time_costs=PEAK_TIME_COSTS, weights=bl_w)
    bl_y = isnothing(bl_shed) ? NaN : meta.fn(meta.use == :served ? bl_served : bl_shed;
                                              peak_time_costs=PEAK_TIME_COSTS, weights=bl_w)

    push!(comparison_df, (criterion, "efficiency_MILP", eff_x,  eff_y))
    push!(comparison_df, (criterion, "fairness_MILP",   fair_x, fair_y))
    push!(comparison_df, (criterion, "bilevel",         bl_x,   bl_y))

    arrow = meta.better == :up ? " — higher = more fair" : " — lower = more fair"
    # Efficiency endpoint uses its own styling; fairness/bilevel use the criterion's.
    crit_mk = FAIR_FUNC_MARKERS[criterion]
    crit_cl = FAIR_FUNC_COLORS[criterion]
    eff_mk = FAIR_FUNC_MARKERS["efficiency"]
    eff_cl = FAIR_FUNC_COLORS["efficiency"]
    p = scatter([eff_x, fair_x, bl_x], [eff_y, fair_y, bl_y],
        series_annotations = text.(["eff", "fair MILP", "bilevel"], :bottom, 8),
        markersize = 9,
        marker = [eff_mk crit_mk crit_mk],
        color = [eff_cl crit_cl :white],
        markerstrokecolor = [eff_cl crit_cl crit_cl],
        xlabel = "Σ λ·Σ pshed (↓ better)",
        ylabel = meta.label,
        title = "$(FAIR_FUNC_LABELS[criterion])" * arrow,
        legend = false,
    )
    push!(comparison_plots, p)
end

comparison_fig = plot(comparison_plots...; layout=(2, 2), size=(1000, 800),
                  plot_title = "Shed vs fairness comparison (3 points per criterion)")
savefig(comparison_fig, joinpath(outdir, "shed_vs_fairness_comparison.svg"))
display(comparison_fig)

CSV.write(joinpath(outdir, "shed_vs_fairness_comparison.csv"), comparison_df)

