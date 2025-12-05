using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS, Juniper
using HSL_jll
using Plots
using Random
using Distributions
using DiffOpt
using JuMP
using LinearAlgebra,SparseArrays
using PowerPlots
using DataFrames
using CSV

# using DataFrames
ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# To make a bilevel JuMP model, we need to create a BilevelJuMP model here 
juniper = optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>ipopt, "mip_solver"=>highs)

#global solver = ipopt

## Main loop
dir = dirname(@__FILE__)

case = "ieee_13_aw_edit/motivation_b.dss"

casepath = "data/$case"
file = joinpath(dir, "..", casepath)

data = case 
vscale = 1
loadscale = 1   

eng = PowerModelsDistribution.parse_file(file)#, transformations=[PowerModelsDistribution.transform_loops!,PowerModelsDistribution.remove_all_bounds!])

eng["settings"]["sbase_default"] = 1
eng["voltage_source"]["source"]["rs"] *=0
eng["voltage_source"]["source"]["xs"] *=0
eng["voltage_source"]["source"]["vm"] *=vscale

"Ensure use the reduce lines function in Fred's basecase script"
#PowerModelsDistribution.reduce_line_series!(eng)


math = PowerModelsDistribution.transform_data_model(eng)


for (idx, switch) in math["switch"]
    switch["state"] = 1
end
lbs = PowerModelsDistribution.identify_load_blocks(math)
get(eng, "time_series", Dict())

for (i,bus) in math["bus"]

		bus["vmax"][:] .= 1.1
		bus["vmin"][:] .= 0.9
end

# Save for the relaxed version when using nonlinear terms in objective
#add_start_vrvi!(math)

# Ensure the generation from the source bus is less than the max load
# First calculate the total load
#ls_percent = 0. # ensure not inf
served = [] #Dict{Any,Any}()
ls_percent = 1
for (i,gen) in math["gen"]
    if gen["source_id"] == "voltage_source.source"
        pd_phase1=0
        pd_phase2=0
        pd_phase3=0
        qd_phase1=0
        qd_phase2=0
        qd_phase3=0
        for (ind, d) in math["load"]
            # @info d
            # @info d["connections"]
            for (idx, con) in enumerate(d["connections"])
                # @info "Load at connection $(d["connections"][idx]) has pd=$(d["pd"][idx]) and qd=$(d["qd"][idx])"
                if 1 == con# d["connections"] 
                    pd_phase1 += d["pd"][idx]
                    qd_phase1 += d["qd"][idx]
                end
                if 2 == con
                    pd_phase2 += d["pd"][idx]
                    qd_phase2 += d["qd"][idx]
                end 
                if 3 == con
                    pd_phase3 += d["pd"][idx]
                    qd_phase3 += d["qd"][idx]
                end
            end
        end
        gen["pmax"][1] = pd_phase1 * ls_percent
        gen["qmax"][1] = qd_phase1 * ls_percent
        gen["pmax"][2] = pd_phase2 * ls_percent
        gen["qmax"][2] = qd_phase2 * ls_percent
        gen["pmax"][3] = pd_phase3 * ls_percent
        gen["qmax"][3] = qd_phase3 * ls_percent
        gen["pmin"][:] .= 0
        gen["qmin"][:] .= 0
    end
end

# Create the critical load set
critical_load = ["675a"]
#critical_load = ["l4"]
for (i,load) in math["load"]
    if load["name"] in critical_load
        load["critical"] = 1
        load["weight"] = 10
        #println("Load $(load["name"]) at math load node $(i) is critical.")
    else
        load["critical"] = 0
        load["weight"] = 10
        #println("Load $(load["name"]) at math load node $(i) is not critical.")

    end
end

for (switch_id, switch) in enumerate(math["switch"])
    math["switch"][string(switch_id)]["branch_id"] = 0
    for (branch_id, branch) in enumerate(math["branch"])
            if branch[2]["source_id"] == switch[2]["source_id"]
                switch[2]["branch_id"] = branch_id  # Assuming you have this mapping
            end
    end
end

# Manual feasibility test
math["switch"]["1"]["state"] = 0 # Open the switch to force load shedding
math["switch"]["2"]["state"] = 1 # Open the switch to force load shedding
math["switch"]["3"]["state"] = 1 # Open the switch to force load shedding
math["block"] = Dict{String,Any}()
for (block, loads) in enumerate(lbs)
    math["block"][string(block)] = Dict("id"=>block, "state"=>0)
end
pm_ivr_soln = solve_mc_pf(math, IVRUPowerModel, ipopt)
pm_ivr_opf_soln = solve_mc_opf(math, IVRUPowerModel, ipopt)

model_bern = JuMP.Model()
set_attribute(model_bern, "hsllib", HSL_jll)
set_attribute(model_bern, "linear_solver", "ma27")
# Create the variables for the Bernoulli switch states
# Want to create a matrix of switch states i = switch_id, j=switch_set_id

"""
    solve_mld_relaxed(data::Dict{String, Any}; optimizer=Ipopt.Optimizer)

Solve the MLD problem with relaxed integer variables using parsed network data.
Returns the result dictionary with continuous switch variables.
"""
# function solve_mld_relaxed(data::Dict{String, Any}; optimizer=Ipopt.Optimizer)
#     # Build the MLD problem with relaxed integers (NLP formulation)
#     # pm = instantiate_mc_model(
#     #     data, 
#     #     LinDist3FlowPowerModel,
#     #     build_mc_mld_switchable;
#     #     ref_extensions=[FairLoadDelivery.ref_add_load_blocks!]
#     # )
#     # ref = pm.ref[:it][:pmd][:nw][0]
#     # # Set optimizer
#     # set_optimizer(pm.model, optimizer)
#     # set_optimizer_attribute(pm.model, "print_level", 0)
    
#     # # Solve the relaxed problem
#     # optimize!(pm.model)
    
function solve_mld_relaxed(data::Dict{String, Any}; optimizer=Ipopt.Optimizer)
    soln = FairLoadDelivery.solve_mc_mld_switch(data, optimizer)
    # Build the MLD problem with relaxed integers (NLP formulation)
    pm = instantiate_mc_model(
        data, 
        LinDist3FlowPowerModel,
        build_mc_mld_switchable;
        ref_extensions=[FairLoadDelivery.ref_add_load_blocks!]
    )
    ref = pm.ref[:it][:pmd][:nw][0]
    # Set optimizer
    set_optimizer(pm.model, optimizer)
    set_optimizer_attribute(pm.model, "print_level", 0)
    # Solve the relaxed problem
    optimize!(pm.model)

    # Extract results
    result = Dict{String, Any}()
    # result["termination_status"] = termination_status(pm.model)
    # result["objective"] = objective_value(pm.model)
    # result["solve_time"] = solve_time(pm.model)
    result["switch"] = soln["solution"]["switch"]

    # Extract switch states (relaxed values)
    result["switch_states"] = Dict{Int, Any}()
    for (i, var) in result["switch"]
        result["switch_states"][parse(Int,i)] = var["state"]
    end
    # Store the full power model for potential warm-starting
    return result,ref
end
res,ref = solve_mld_relaxed(math; optimizer=ipopt)
z_relaxed = res["switch_states"]
pm_ivr_soln = solve_mc_pf(math, IVRUPowerModel, ipopt)

"""
    generate_bernoulli_samples(switch_states::Dict{Int, Float64}; 
                               n_samples=10, 
                               seed=nothing)

Generate sets of Bernoulli variables based on relaxed switch probabilities.
Each switch i with relaxed value p_i is sampled as Bernoulli(p_i).
Returns an array of sample dictionaries.
"""

function generate_bernoulli_samples(switch_states::Dict{Int, Any}; 
                                   n_samples=10, 
                                   seed=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    samples = Vector{Dict{Int, Float64}}()
    
    for sample_idx in 1:n_samples
        sample = Dict{Int, Float64}()
        for (switch_id, p) in switch_states
            # Generate Bernoulli random variable with probability p
            sample[switch_id] = Float64(rand(Bernoulli(p)))
        end
        push!(samples, sample)
    end
    
    return samples
end
bernoulli_samples = generate_bernoulli_samples(res["switch_states"]; n_samples=10, seed=42)
optimizer=ipopt
# """
#     optimize_bernoulli_selection(z_relaxed::Dict{Int, Any},
#                                  bernoulli_samples::Vector{Dict{Int, Float64}};
#                                  optimizer=Ipopt.Optimizer)

# Find the single best Bernoulli sample that minimizes distance to the relaxed solution.

# Solves: min ||z_bern_i - z_relaxed||_2^2
#         where i is selected from the set of Bernoulli samples

# This uses continuous relaxation of binary selection variables, then picks
# the sample with highest weight.
# """
reference = ref
function radiality_check(ref_round::Dict{Symbol,Any}, z_relaxed::Dict{Int, Any},
                                      bernoulli_samples::Vector{Dict{Int, Float64}};
                                      optimizer=Ipopt.Optimizer)
    
    n_samples = length(bernoulli_samples)
    n_switches = length(z_relaxed)
    switch_ids = sort(collect(keys(z_relaxed)))
    
    println("\n[Optimization] Finding best single Bernoulli sample...")
    println("  Number of samples: $n_samples")
    println("  Number of switches: $n_switches")
    
   
    # Create variables for the rounded switch states
    #bernoulli_selection = JuMP.@variable(model_ran, y[1:n_samples], Bin)
    bernoulli_selection_exp = Vector{Dict{Int, JuMP.VariableRef}}()
    #bernoulli_selection_val = Any[]# Vector{Dict{Int, Float64}}()

     # Calculate L2 distance for each sample
        distances = Any[]
        radial_cons = Any[]
        math_round = deepcopy(math)
       
    for i in 1:n_samples
        # Build a model to evaluate distances
        model_ran = JuMP.Model()
        set_attribute(model_ran, "hsllib", HSL_jll.libhsl_path)
        set_attribute(model_ran, "linear_solver", "ma27")
        # Set optimizer
        set_optimizer(model_ran, optimizer)
        set_optimizer_attribute(model_ran, "print_level", 0)
        bernoulli_switch = Vector{Dict{Int, JuMP.VariableRef}}()
        
        for sample_idx in 1:n_samples
            switch_state = Dict{Int, JuMP.VariableRef}()
            for s in switch_ids
                switch_state[s] = JuMP.@variable(model_ran, base_name="switch_$(sample_idx)_$(s)", lower_bound=0.0, upper_bound=1.0)  # Relaxed to [0,1]
            end
            push!(bernoulli_switch, switch_state)
        end

        # Very safe M (switches ∈ [0,1])
        # M = 1.0
        # M_radial = 1e4

        distance = @expression(model_ran, sum((bernoulli_switch[i][s] - z_relaxed[s])^2 for s in switch_ids))
        for s in switch_ids
            #math_round["switch"][string(s)]["state"] = bernoulli_samples[i][s]   
            # @constraint(model_ran,
            #     bernoulli_switch[i][s] == bernoulli_samples[i][s] <= M * (1 - y[i]))            
            # @constraint(model_ran,
            #     bernoulli_switch[i][s] - bernoulli_samples[i][s] >= -M * (1 - y[i]))            
            JuMP.@constraint(model_ran, bernoulli_switch[i][s] == bernoulli_samples[i][s])
        end
        JuMP.@constraint(model_ran, sum(bernoulli_switch[i][s] for s in switch_ids) >= 1)  # At least one switch closed
        #FairLoadDelivery.constraint_rounded_switch_states(model_ran,ref_round,switch_state)
	    FairLoadDelivery.constraint_radial_topology_jump(model_ran,ref_round,bernoulli_switch[i];bern=false)
        #push!(radial_cons, radial_con)
        JuMP.@objective(model_ran, Min, 0)
        optimize!(model_ran)
        term_status = JuMP.termination_status(model_ran)
        if term_status == MOI.OPTIMAL || term_status == MOI.LOCALLY_SOLVED
            push!(bernoulli_selection_exp, bernoulli_switch[i])
            #push!(bernoulli_selection_val, value.(bernoulli_switch[i][s] for s in switch_ids))
            push!(distances, distance)
        else
           # println("Optimization did not converge to optimality for sample $i. Status: $term_status")
        end
    end

    # Find the sample with minimum distance
    # Create an objective to minimize distance
    # JuMP.@objective(model_ran, Min, sum(distances[i] for i in 1:n_samples))
    # optimize!(model_ran)
    # Extract best sample info
    # best_sample_idx = argmin(value.(distances))
    # best_distance = distances[best_sample_idx]
    return bernoulli_selection_exp, switch_ids
end
bernoulli_selection,switch_ids = radiality_check(reference, z_relaxed,
                                      bernoulli_samples;
                                      optimizer=ipopt)    
if length(bernoulli_selection) == 0
    error("No radial solutions found among Bernoulli samples.")
else
    println("Number of radial solutions from Bernoulli samples: $(length(bernoulli_selection))")
end
# Test the acpf feasibility for the new set of switches that passed radiality
function ac_feasibility_test(math::Dict{String, Any}, 
                          bernoulli_selection_exp::Any,
                          switch_ids::Vector{Int};
                          optimizer=Ipopt.Optimizer)
    ac_bernoulli = Any[]
    ac_bernoulli_val = Any[]# Vector{Dict{Int, Float64
    math_round = deepcopy(math)
    for (set_id, bernoulli_set) in enumerate(bernoulli_selection_exp)
        for s in switch_ids
            math_round["switch"][string(s)]["state"] = value(bernoulli_set[s])
        end
        #pm_ivr_soln_round = solve_mc_pf(math_round, IVRUPowerModel, optimizer)
        pf_ivrup_aw = solve_mc_pf_aw(math_round, ipopt)
        term_status = pf_ivrup_aw["termination_status"]
        if term_status == MOI.OPTIMAL || term_status == MOI.LOCALLY_SOLVED
            push!(ac_bernoulli,bernoulli_set)
            push!(ac_bernoulli_val, value.(bernoulli_set[s] for s in switch_ids))
        else
           # println("Optimization did not converge to optimality for sample $set_id. Status: $term_status")
        end
    end
    return ac_bernoulli, ac_bernoulli_val
end
ac_bernoulli,ac_bernoulli_val = ac_feasibility_test(math, bernoulli_selection, switch_ids; optimizer=ipopt)
if length(ac_bernoulli) == 0
    error("No feasible AC solutions found among Bernoulli samples.")
else
    println("Number of feasible AC solutions from Bernoulli samples: $(length(ac_bernoulli))")
end


# Find the set of switches with best mld objective value among feasible ac solutions
function find_best_switch_set(math::Dict{String, Any}, 
                          ac_bernoulli::Vector{Any},
                          switch_ids::Vector{Int};
                          optimizer=Ipopt.Optimizer)
    best_obj = -Inf
    best_sample_idx = 0
    best_switch_config = Dict{Int, Float64}()
    math_round = deepcopy(math)
    for (set_id, bernoulli_set) in enumerate(ac_bernoulli)
        for s in switch_ids
            math_round["switch"][string(s)]["state"] = value(bernoulli_set[s])
        end
        mld_rounded_soln = solve_mc_mld_shed_random_round(math_round, optimizer)
        obj_val = mld_rounded_soln["objective"]
        term_status = mld_rounded_soln["termination_status"]
        @info "Sample $set_id: Objective = $obj_val, Status = $term_status"
        if (term_status == MOI.OPTIMAL || term_status == MOI.LOCALLY_SOLVED) && obj_val > best_obj
            best_obj = obj_val
            best_sample_idx = set_id
            for s in switch_ids
                best_switch_config[s] = value(bernoulli_set[s])
            end
        end
    end

    return best_sample_idx, best_switch_config
end

best_sample_idx, best_switch_config = find_best_switch_set(math, ac_bernoulli, switch_ids; optimizer=ipopt)
if best_sample_idx == 0
    error("No feasible MLD solutions found among AC feasible Bernoulli samples.")
end
println("Best feasible switch configuration found at sample index: $best_sample_idx")
println("Best switch configuration: $best_switch_config")
#Set the math dictionary to use the best switch configuration
math_fin = deepcopy(math)
for s in switch_ids
    @info "Setting switch $s to state $(best_switch_config[s])"
    math_fin["switch"][string(s)]["state"] = best_switch_config[s]
end
# Solve the final MLD problem with the best switch configuration
mld_rounded_soln = solve_mc_mld_shed_random_round(math_fin, optimizer)
##############################################
# Extract results
##############################################
res = mld_rounded_soln["solution"]
#res = pm_mld_soln["solution"]
# println("Load served: $(sum(load["pd"] for load in math["load"] if load["critical"] == 1))")
#load_ref = sum(load["pd"][idx] for (idx, con) in enumerate(load["connections"]) for (i,load) in ref[:load] )
load_ref = []
for (i, load) in sort(ref[:load])
    cons = load["connections"]
    for idx in 1:length(cons)
        push!(load_ref, load["pd"][idx])
    end
end
load_ref_sum = sum(load_ref)
println("Total load in reference: $load_ref_sum")

gen_ref = []# sum(gen["pg"] for (i,gen) in ref[:gen])
for (i, gen) in ref[:gen]
    cons = gen["connections"]
    for idx in 1:length(cons)
        push!(gen_ref, gen["pg"][idx])
    end
end
gen_ref_sum = sum(gen_ref)
println("Total generation in reference: $gen_ref_sum")

gen_soln = []# sum(gen["pg"] for (i,gen) in ref[:gen])
for (i, gen) in res["gen"]
    for idx in 1:length(gen["pg"])
        push!(gen_soln, gen["pg"][idx])
    end
end
gen_soln_sum = sum(gen_soln)
println("Total generation in solution: $gen_soln_sum")

#load_served = sum((load["pd"]) for (i,load) in res["load"])
load_served = []
idxs = sort(parse.(Int,collect(keys(res["load"]))))
for i in 1:length(idxs)
    load = res["load"][string(i)]
    for idx in 1:length(load["pd"])
        push!(load_served, load["pd"][idx])
    end
end
load_served_sum = sum(load_served)

println("Total load served in MLD solution: $load_served_sum")
println("Load served percentage: $(load_served_sum/load_ref_sum*100) %")
push!(served, (load_served_sum/load_ref_sum)*100)
#end
#println(served)

# Print the following fairness indices: Gini index, Jain's index, Palma Ratio, Alpha fairness for alpha=1
#Gini index
function gini_index(x)
    x = sort(x)
    n = length(x)
    #mean_x = _PMD.mean(x)
    #sum_diff = sum(abs(x[i] - x[j]) for i in 1:n for j in 1:n)
    gini_top = 1 - 1/n + 2*sum(sum(x[j] for j in 1:i) for i in 1:n-1)/(n*sum(x))
    gini_bottom = 2*(1-1/n)
    return gini_top/gini_bottom
end

#Jain's index
function jains_index(x)
    n = length(x)
    sum_x = sum(x)
    sum_x2 = sum(xi^2 for xi in x)
    return (sum_x^2) / (n * sum_x2)
end

#Palma Ratio
function palma_ratio(x)
    x = load_served ./ load_ref
    sorted_x = sort(x)
    n = length(x)
    top_10_percent = sum(sorted_x[ceil(Int, 0.9n):end])
    bottom_40_percent = sum(sorted_x[1:floor(Int, 0.4n)])
    return top_10_percent / bottom_40_percent
end
#Alpha fairness for alpha=1
function alpha_fairness(x, alpha=1)
if alpha == 1
    return sum(log(xi) for xi in x)
else
    return sum((xi^(1 - alpha)) / (1 - alpha) for xi in x)
end
end

# Calculate and print fairness indices
served_array = collect(load_served./load_ref)
println("Gini Index: ", gini_index(served_array))
println("Jain's Index: ", jains_index(served_array))    
println("Palma Ratio: ", palma_ratio(served_array))
println("Alpha Fairness (alpha=1): ", alpha_fairness(served_array, 1))
println("Alpha Fairness (alpha=0.5): ", alpha_fairness(served_array, 0.5))
println("Alpha Fairness (alpha=5): ", alpha_fairness(served_array, 5))

# Make the fairness results into a dataframe and save as csv

fairness_df = DataFrame(Gini_Index=gini_index(served_array), Jains_Index=jains_index(served_array), Palma_Ratio=palma_ratio(served_array), Alpha_Fairness_1=alpha_fairness(served_array, 1), Alpha_Fairness_0_5=alpha_fairness(served_array, 0.5), Alpha_Fairness_5=alpha_fairness(served_array, 5))
#fairness_df.Alpha_Fairness_5 = alpha_fairness(served_array, 5)
CSV.write("fairness_results_gini.csv", fairness_df)



# return Dict(
    #     "best_sample_idx" => best_sample_idx,
    #     "best_sample" => best_switch_config,
    #     "best_distance" => value(best_distance)
    # )



    # """
#     solve_mld_fixed_switches(data::Dict{String, Any}, 
#                             switch_states::Dict{Int, Float64};
#                             optimizer=Ipopt.Optimizer)

# Solve the MLD problem with fixed switch states (pure NLP) using parsed network data.
# """
# function solve_mld_fixed_switches(data::Dict{String, Any}, 
#                                   switch_states::Dict{Int, Float64};
#                                   optimizer=Ipopt.Optimizer)
#     # Build the MLD problem
#     pm = instantiate_mc_model(
#         data, 
#         IVRUPowerModel,
#         build_mc_mld;
#         ref_extensions=[ref_add_connected_components!]
#     )
    
#     # Fix switch variables to rounded values
#     for (i, var) in pm.model[:z_switch]
#         fix(var, switch_states[i]; force=true)
#     end
#     model_ran = JuMP.Model()
#     set_attribute(model_ran, "hsllib", HSL_jll)
#     set_attribute(model_ran, "linear_solver", "ma27")
#     # Set optimizer
#     set_optimizer(model_ran, optimizer)
#     set_optimizer_attribute(model_ran, "print_level", 0)

#     # Solve the fixed problem
#     optimize!(model_ran)
    
#     # Extract results
#     result = Dict{String, Any}()
#     result["termination_status"] = termination_status(pm.model)
#     result["objective"] = objective_value(pm.model)
#     result["solve_time"] = solve_time(pm.model)
#     result["switch_states"] = switch_states
    
#     return result
# end

# """
#     bernoulli_optimization_mld(data::Dict{String, Any}; 
#                               n_samples=10,
#                               seed=nothing,
#                               optimizer=Ipopt.Optimizer)

# Complete Bernoulli optimization procedure using parsed network data:
# 1. Solve relaxed MLD to get continuous switch values z_relaxed
# 2. Generate n_samples sets of Bernoulli variables based on z_relaxed probabilities
# 3. Find the single best Bernoulli sample that minimizes L2 distance to z_relaxed
# 4. Evaluate that sample on actual MLD problem
# """
# function bernoulli_optimization_mld(data::Dict{String, Any}; 
#                                    n_samples=10,
#                                    seed=nothing,
#                                    optimizer=Ipopt.Optimizer)
#     println("=" ^ 60)
#     println("Bernoulli Optimization MLD Procedure")
#     println("=" ^ 60)
    
#     # Step 1: Solve relaxed problem
#     println("\n[Step 1] Solving relaxed MLD problem...")
#     relaxed_result,ref = solve_mld_relaxed(data; optimizer=optimizer)
    
#     println("  Status: $(relaxed_result["termination_status"])")
#     println("  Objective: $(round(relaxed_result["objective"], digits=4))")
    
#     # Display relaxed switch values
#     println("\n  Relaxed switch values (z_relaxed):")
#     for (sw_id, val) in sort(collect(relaxed_result["switch_states"]))
#         println("    Switch $sw_id: $(round(val, digits=4))")
#     end
    
#     # Step 2: Generate Bernoulli samples
#     println("\n[Step 2] Generating $n_samples Bernoulli samples...")
#     bernoulli_samples = generate_bernoulli_samples(
#         relaxed_result["switch_states"]; 
#         n_samples=n_samples, 
#         seed=seed
#     )
    
#     println("  Generated $(length(bernoulli_samples)) samples")
#     println("\n  Sample preview (first 3 samples):")
#     for i in 1:min(20, n_samples)
#         println("    Sample $i: ", join(["S$s=$(Int(v))" for (s,v) in sort(collect(bernoulli_samples[i]))], ", "))
#     end
    
#     # Step 3: Find best single sample (minimize L2 distance)
#     println("\n[Step 3] Finding best Bernoulli sample...")
#     selection_result = optimize_bernoulli_selection(ref,
#         relaxed_result["switch_states"],
#         bernoulli_samples;
#         optimizer=optimizer
#     )
    
#     best_switches = selection_result["best_sample"]
    
#     # Display selected sample
#     println("\n  Selected binary solution (Sample $(selection_result["best_sample_idx"])):")
#     for (sw_id, val) in sort(collect(best_switches))
#         relaxed_val = relaxed_result["switch_states"][sw_id]
#         println("    Switch $sw_id: $(Int(val)) (relaxed: $(round(relaxed_val, digits=4)))")
#     end
    
#     # Step 4: Evaluate best sample on actual MLD problem
#     println("\n[Step 4] Evaluating binary solution on MLD problem...")
#     try
#         final_result = solve_mld_fixed_switches(
#             data,
#             best_switches;
#             optimizer=optimizer
#         )
        
#         println("  Status: $(final_result["termination_status"])")
#         println("  Objective: $(round(final_result["objective"], digits=4))")
        
#         println("\n" * "=" ^ 60)
#         println("Results Summary")
#         println("=" ^ 60)
#         println("Relaxed objective:         $(round(relaxed_result["objective"], digits=4))")
#         println("Binary solution objective: $(round(final_result["objective"], digits=4))")
#         println("Optimality gap:            $(round((relaxed_result["objective"] - final_result["objective"]) / relaxed_result["objective"] * 100, digits=2))%")
#         println("Selected sample:           #$(selection_result["best_sample_idx"])")
#         println("L2 distance² (to relaxed): $(round(selection_result["best_distance"], digits=6))")
#         println("=" ^ 60)
        
#         return Dict(
#             "relaxed_result" => relaxed_result,
#             "bernoulli_samples" => bernoulli_samples,
#             "selection_result" => selection_result,
#             "final_result" => final_result
#         )
#     catch e
#         println("  Error evaluating solution: $e")
#         return Dict(
#             "relaxed_result" => relaxed_result,
#             "bernoulli_samples" => bernoulli_samples,
#             "selection_result" => selection_result,
#             "error" => e
#         )
#     end
# end

# # Example usage:
# # Parse network data first
# # data = parse_file("path/to/network.dss")
# results = bernoulli_optimization_mld(math; n_samples=20, seed=42)
