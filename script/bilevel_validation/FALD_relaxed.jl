
include("ExperimentSetup.jl")

math_new = deepcopy(math)
fair_weights = Float64[]
for (load_id, load_dict) in math["load"]
    push!(fair_weights, load_dict["weight"])
end
    
pshed_lower_level = Float64[]
pshed_upper_level = Float64[]
final_weight_ids = Int[]
final_weights = Float64[]

for k in 1:ITERATIONS
    println("\n  --- Iteration $k ---")
    # Solve lower-level problem and get sensitivities
    dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, _ = lower_level_soln(math_new, fair_weights, 1);
    # Apply fairness function
    if fair_func == "proportional"
        pshed_new, fair_weight_vals = proportional_fairness_load_shed(dpshed, pshed_val, weight_vals, math_new);
    elseif fair_func == "efficiency"
        pshed_new, fair_weight_vals = complete_efficiency_load_shed(dpshed, pshed_val, weight_vals, math_new);
    elseif fair_func == "min_max"
        pshed_new, fair_weight_vals = min_max_load_shed(dpshed, pshed_val, weight_vals);
    elseif fair_func == "equality_min"
        pshed_new, fair_weight_vals = FairLoadDelivery.equality_min(dpshed, pshed_val, weight_vals);
    elseif fair_func == "jain"
        pshed_new, fair_weight_vals = jains_fairness_index(dpshed, pshed_val, weight_vals);
    elseif fair_func == "palma"
        pd = Float64[]
        for (load_id, load_dict) in math_new["load"]
            push!(pd, sum(load_dict["pd"]))
        end
        pshed_new, fair_weight_vals = lin_palma_reformulated(dpshed, pshed_val, weight_vals, pd);
    else
        error("Unknown fairness function: $fair_func")
    end

    # Update weights in math dict
    for (i, w) in zip(weight_ids, fair_weight_vals)
        math_new["load"][string(i)]["weight"] = w
    end

    push!(pshed_lower_level, sum(pshed_val))
    push!(pshed_upper_level, sum(pshed_new))

    println("    Lower-level shed: $(sum(pshed_val)), Upper-level shed: $(sum(pshed_new))")
    println("    Weights: $fair_weight_vals, with type: $(typeof(fair_weight_vals))")
end

mld_relaxed_final = FairLoadDelivery.solve_mc_mld_shed_implicit_diff(math_new, ipopt_solver; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!]);
math_relaxed = deepcopy(math_new);