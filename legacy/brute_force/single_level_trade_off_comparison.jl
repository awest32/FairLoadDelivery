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

# Load the test case
case = "pmd_opendss/case3_balanced_battery.dss"#"ieee_13_aw_edit/motivation_c.dss"
dir = joinpath(@__DIR__,"../", "data")
file = joinpath(dir, case)

# Test parameters
gen_cap = 0.8
source_pu = 1.03
switch_rating = 600.0
critical_loads = []

# Load the data
eng, math = setup_network(file, gen_cap, source_pu, switch_rating, critical_loads)

#result = solve_mc_opf(eng, LinDist3FlowPowerModel, Ipopt.Optimizer)
efficient_soln_int = solve_mc_mld_switch_integer(math, Gurobi.Optimizer)


# TODO add comparison with single level fairness trade-off parameter for each fairness function, to see how the upper-level solution with fixed inputs from the lower level compares to the single-level solutions.
jain_soln = solve_mc_mld_jain_integer(math, Gurobi.Optimizer)
min_max_soln = solve_mc_mld_min_max_integer(math, Gurobi.Optimizer)
proportional_soln = solve_mc_mld_proportional_fairness_integer(math, Gurobi.Optimizer)
efficient_soln = solve_mc_mld_switch_integer(math, Gurobi.Optimizer)
palma_soln = solve_mc_mld_palma_integer(math, Gurobi.Optimizer)

# Okay compare the most efficient solution with the other solutions in terms of load shed 
# make a plot of the load shed for each solution, and the fairness weights for each solution, to see how they compare.
jain_pserved = sum(sum(jain_soln["solution"]["load"][string(i)]["pd"]) for i in keys(jain_soln["solution"]["load"]))
min_max_pserved = sum(sum(min_max_soln["solution"]["load"][string(i)]["pd"]) for i in keys(min_max_soln["solution"]["load"]))
proportional_pserved = sum(sum(proportional_soln["solution"]["load"][string(i)]["pd"]) for i in keys(proportional_soln["solution"]["load"]))
efficient_pserved = sum(sum(efficient_soln["solution"]["load"][string(i)]["pd"]) for i in keys(efficient_soln["solution"]["load"]))
palma_pserved = sum(sum(palma_soln["solution"]["load"][string(i)]["pd"]) for i in keys(palma_soln["solution"]["load"]))
pserved_df = DataFrame(
    fairness_function = ["jain", "min_max", "proportional", "efficient", "palma"],
    load_shed = [jain_pserved, min_max_pserved, proportional_pserved, efficient_pserved, palma_pserved]
)
@df pserved_df bar(:fairness_function, :load_shed, title="Total Load Shed for Each Fairness Function")
xlabel!("Fairness Function")
ylabel!("Total Load Shed (kW)")

#-----------------------------------------------------
# Bilevel results with current approach
#------------------------------------------------------
# Solve the lower level problem
# Get initial weights
fair_weights_init = Float64[]
for (load_id, load) in math["load"]
    push!(fair_weights_init, load["weight"])
end

iterations = 3
mn_new = deepcopy(math)
fair_weights = copy(fair_weights_init)  # N-length global weights for lower level
n_weights = length(fair_weights)
nw_ids = 1#sort(collect(keys(mn_new["nw"])), by=x->parse(Int, x))
T = length(nw_ids)

pshed_lower_level = Float64[]
pshed_upper_level = Float64[]
final_weight_ids = Int[]
final_weights = Float64[]
completed_iterations = 0
last_status = MOI.OPTIMIZE_NOT_CALLED
prev_weights = repeat(fair_weights, T)  # T*N for per-period tracking
prev_pshed = Float64[]
max_delta_weights = NaN
max_delta_pshed = NaN
iteration_log = []  # Per-iteration diagnostics
peak_time_costs = 1# Example: uniform peak time costs
for fair_func in ["proportional", "efficiency", "min_max", "jain", "palma"]
    for k in 1:iterations
        # Solve multiperiod lower-level and get (T*N) x (T*N) Jacobian
        dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs = lower_level_soln_mn(mn_new, fair_weights, k)

        n_loads = length(weight_ids) # N (loads per period)

        # Collect per-period pd reference values matching pshed ordering from Jacobian
        pd_all = Float64[]
        for (nw, lid) in pshed_nw_ids
            push!(pd_all, sum(refs[nw][:load][lid]["pd"]))
        end

        # Apply fairness function on the full per-period (T*N) pshed vector
        # dpshed is (T*N) x (T*N), pshed_val is (T*N), weight_vals is (T*N)
        if fair_func == "proportional"
            pshed_new, fair_weight_vals, status = proportional_fairness_load_shed(dpshed, pshed_val, weight_vals, pd_all)
        elseif fair_func == "efficiency"
            pshed_new, fair_weight_vals, status = efficient_load_shed(dpshed, pshed_val, weight_vals)
        elseif fair_func == "min_max"
            pshed_new, fair_weight_vals, status = min_max_load_shed(dpshed, pshed_val, weight_vals)
        elseif fair_func == "equality_min"
            pshed_new, fair_weight_vals, status = equality_min(dpshed, pshed_val, weight_vals)
        elseif fair_func == "jain"
            pshed_new, fair_weight_vals, status = jains_fairness_index(dpshed, pshed_val, weight_vals)
        elseif fair_func == "palma"
            pshed_new, fair_weight_vals, status = lin_palma_reformulated(dpshed, pshed_val, weight_vals, pd_all)
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

        # Track convergence (compare per-period weights)
        if k!=1
            max_delta_weights = maximum(abs.(fair_weight_vals .- prev_weights))
        end

        if !isempty(prev_pshed)
            max_delta_pshed = maximum(abs.(pshed_new .- prev_pshed))
        end
        prev_weights = copy(fair_weight_vals)
        prev_pshed = copy(pshed_new)

        # Update weights per-period in mn_data
        for (t, nw_id) in enumerate(nw_ids)
            nw_data = mn_new
            offset = (t - 1) * n_loads
            for (j, lid) in enumerate(weight_ids)
                nw_data["load"][string(lid)]["weight"] = fair_weight_vals[offset + j]
            end
        end

        # Pass T*N per-period weights to next lower-level iteration
        fair_weights = copy(fair_weight_vals)
        final_weight_ids = weight_ids
        final_weights = fair_weight_vals

        push!(pshed_lower_level, sum(pshed_val))
        push!(pshed_upper_level, sum(pshed_new))

        # Log iteration diagnostics
        dw = fair_weight_vals .- prev_weights
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
            n_pshed_total = length(pshed_val)
        ))
    end
end