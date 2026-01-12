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
using Plots

include("../src/implementation/network_setup.jl")
include("../src/implementation/lower_level_mld.jl")
include("../src/implementation/palma_relaxation.jl")
include("../src/implementation/other_fair_funcs.jl")
include("../src/implementation/random_rounding.jl")
include("../src/implementation/export_results.jl")

ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# Inputs: case file path, percentage of load shed, list of critical load IDs
ls_percent = 0.9
eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/motivation_b.dss", ls_percent, [])

#Initial fair load weights
fair_weights = Float64[]
for (load_id, load) in (math["load"])
    push!(fair_weights, load["weight"])
end
          
dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, mld_soln  = lower_level_soln(math, fair_weights, 1)

# Make a copy of the math dictionary
math_new = deepcopy(math)


iterations = 10
#fair_func = "equality_min" # Options: "proportional", "efficiency", "min_max", "jain", "palma"
case = "base_motivation_b"

# Function to run the relaxed PADP (formally FLDP) for a given number of iterations
function relaxed_fldp(data::Dict{String, Any}, iterations::Int, fair_weights::Vector{Float64}, fair_func::String, save_path::String)
    # Build and solve the relaxed Priority Aware Delivery Problem (PADP)
    ref_out = []
    pshed_lower_level = []
    pshed_upper_level = []
    pshed_vals_out = Float64[]
    pshed_ids_out = Int64[]
    weight_vals = Float64[]
    weight_ids_out = Int64[]
    for k in 1:iterations
        @info "Starting iteration $k"

        # Solve lower-level problem and get sensitivities
        dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, lower_level_ref = lower_level_soln(data, fair_weights, k)
        push!(ref_out,lower_level_ref)
        # Plot the heatmap of sensitivities
        plot_dpshed_heatmap(dpshed, pshed_ids, weight_ids, k, save_path)
        
        # Plot load shed per bus
        plot_load_shed_per_bus(pshed_val, pshed_ids, k, save_path)
        
        # Update weights using Lin-PALMA-W with gradient input
        # pshed_new, fair_weight_vals, sigma = lin_palma_w_grad_input(dpshed, pshed_val, weight_vals, pd)
        if fair_func == "proportional"
            pshed_new, fair_weight_vals = proportional_fairness_load_shed(dpshed, pshed_val, weight_vals)
        elseif fair_func == "efficiency"
            pshed_new, fair_weight_vals = complete_efficiency_load_shed(dpshed, pshed_val, weight_vals)
        elseif fair_func == "min_max"
            pshed_new, fair_weight_vals = min_max_load_shed(dpshed, pshed_val, weight_vals) 
        elseif fair_func == "equality_min"
            pshed_new, fair_weight_vals = equality_min(dpshed, pshed_val, weight_vals)
        elseif fair_func == "jain"
            pshed_new, fair_weight_vals = jains_fairness_index(dpshed, pshed_val, weight_vals)
        elseif fair_func == "palma"
             # Order the load using the indices from the pshed_ids
            pd = Float64[]
            for i in pshed_ids
                push!(pd, sum(math_new["load"][string(i)]["pd"]))
            end
            pshed_new, fair_weight_vals = lin_palma_w_grad_input(dpshed, pshed_val, weight_vals, pd)
        end
        
        plot_weights_per_load(fair_weight_vals, weight_ids, k, save_path)
        
        # Update the fair load weights in the math dictionary
        for (i, w) in zip(weight_ids, fair_weight_vals)
            math_new["load"][string(i)]["weight"] = w
        end
        
        # Store the total load shed for this iteration
        push!(pshed_lower_level, sum(pshed_val))
        push!(pshed_upper_level, sum(pshed_new))
        weight_ids_out = weight_ids
        weight_vals = fair_weight_vals
        pshed_vals_out = pshed_val
        pshed_ids_out = pshed_ids
    end
    return math, pshed_lower_level, pshed_upper_level, pshed_vals_out, pshed_ids_out, weight_vals, weight_ids_out, ref_out[1]
end
# Voltage source consistency check
function voltage_source_check(math_og::Dict{String,Any}, math_changed::Dict{String,Any})
    for (i,gen) in math_changed["gen"]
        if gen["source_id"] == "voltage_source.source"
            for (idx, v) in enumerate(gen["vg"])
                if v != math_og["gen"][i]["vg"][idx]
                    @error "Voltage source generation voltage inconsistent after relaxation!"
                else
                    @info "Voltage source generation voltage consistent after relaxation."
                end
            end
            if gen["vbase"] != math_og["gen"][i]["vbase"]
                @error "Voltage source generation vbase inconsistent after relaxation!"
            else
                @info "Voltage source generation vbase consistent after relaxation."
            end
        end
    end
end
function voltage_source_check(math_og::Dict{String,Any}, ref_changed::Dict{Symbol,Any})
    for (i,gen) in ref_changed[:gen]
        if gen["source_id"] == "voltage_source.source"
            for (idx, v) in enumerate(gen["vg"])
                if v != math_og["gen"][string(i)]["vg"][idx]
                    @error "Voltage source generation voltage inconsistent after relaxation!"
                else
                    @info "Voltage source generation voltage consistent after relaxation."
                end
            end
            if gen["vbase"] != math_og["gen"][string(i)]["vbase"]
                @error "Voltage source generation vbase inconsistent after relaxation!"
            else
                @info "Voltage source generation vbase consistent after relaxation."
            end
        end
    end
end
# Extract the switch states and load block statuses from the math dictionary, save as seperate dictionaries 
function extract_switch_block_states(relaxed_soln::Dict{String,Any})
    # Create the switch state dictionary
    switch_states = Dict{Int64,Float64}([])
    for (s_id, s_data) in relaxed_soln["switch"]
        switch_states[parse(Int,s_id)] = s_data["state"]
    end

    #Create the block status dictionary
    block_status = Dict{Int64,Float64}([])
    for (b_id, b_data) in relaxed_soln["block"]
        block_status[parse(Int,b_id)] = b_data["status"]
    end
    return switch_states, block_status
end
# Find the AC voltage violations from the AC feasibility test results
function count_voltage_violations(math_out, ac_feas)
    v_viol_over = []
    v_viol_under = []
    bus_con_num = 0
    overvoltage_violation_num = 0
    undervoltage_violation_num = 0

    for (id, data) in enumerate(math_out)
        v_squared_max = 1.1^2
        v_squared_min = 0.9^2
        feas = ac_feas[id]

        for (bus_id, bus) in pairs(data["bus"])
            v_sq = feas["solution"]["bus"][bus_id]["vr"].^2 +
                   feas["solution"]["bus"][bus_id]["vi"].^2

            for ws in v_sq
                bus_con_num += 1
                if ws > v_squared_max
                    push!(v_viol_over, (id, bus_id, ws, "overvoltage"))
                    overvoltage_violation_num += 1
                elseif ws < v_squared_min
                    push!(v_viol_under, (id, bus_id, ws, "undervoltage"))
                    undervoltage_violation_num += 1
                end
            end
        end
    end

    return bus_con_num, overvoltage_violation_num, undervoltage_violation_num,
           v_viol_over, v_viol_under
end

# Find the feasibility solution which serves the most load
function export_ac_feasibility_results(math_out::Vector{Dict{String, Any}}, ac_feas::Vector{Dict{String, Any}}, save_path::String)
    max_load_served = -Inf
    best_feasibility = nothing
    for feas in ac_feas
        if haskey(feas, "feas_obj") && feas["feas_obj"] != nothing
        # load_served = feas["feas_obj"]
        pd_served = 0.0
        qd_served = 0.0
        for (load_id, load_data) in feas["solution"]["load"]
                pd_served += sum(load_data["pd"])
                qd_served += sum(load_data["qd"])
        end
            if pd_served > max_load_served
                max_load_served = pd_served
                best_feasibility = feas
            end
            @info "Maximum load served among feasible ACPF solutions: $max_load_served"
            if best_feasibility != nothing
            # @info "Best feasibility solution details: $best_feasibility"        
            end
        else
            @warn "Feasibility dictionary does not contain an ACPF feasible solution."
            @info "Finding best MLD objective"
        end
    end
    return best_feasibility, max_load_served
end


function find_best_mld_solution(math_out::Vector{Dict{String, Any}}, ipopt)
    best_obj = -Inf
    best_set = 0
    best_mld = Dict{String, Any}()
    for (id, data) in enumerate(math_out)
        mld = FairLoadDelivery.solve_mc_mld_shed_random_round(data, ipopt)
        @info "Rounded solution from set $id has termination status: $(mld["termination_status"]) and objective value: $(mld["objective"])"
        if best_obj <= mld["objective"] 
            best_obj = mld["objective"]
            best_set = id
            best_mld = mld
        end
    end
    return best_set, best_mld
end

# Run the relaxed FLDP for each fairness function and created two plots: 1) total load shed final output for each function (labels) vs. load index 
# 2) weight values final output for each function (labels) vs. load index
# 3) table of the fairness metric value for each function
relaxed_fldp_pshed_plot = plot()
relaxed_fldp_weight_plot = plot()
relaxed_fldp_fairness_table = DataFrame(FairnessFunction=String[], MinMax=Float64[], Proportional=Float64[], Jain=Float64[], Palma=Float64[], Gini=Float64[])
for fair_func in ["proportional", "efficiency", "min_max", "jain", "equality_min"]#"palma"]
    @info "Running relaxed FLDP with $fair_func fairness function"
   
    save_path = "results/$(Dates.today())/$case/$fair_func/"
    create_save_folder(case, fair_func)
    math_relaxed, pshed_lower_level, pshed_upper_level, pshed_val, pshed_ids, weight_vals, weight_ids_fin, model = relaxed_fldp(math_new, iterations, fair_weights, fair_func, save_path)

    # # Check the results of the relaxed FLDP
    # mld_relaxed = FairLoadDelivery.solve_mc_mld_switch_relaxed(math_relaxed, ipopt)
    # math_rounded = FairLoadDelivery.solve_mc_mld_shed_random_round(math_relaxed, ipopt)

    voltage_source_check(math, math_relaxed)

    # Save and plot the results of the relaxed FLDP
    export_results(math_relaxed, save_path, sum(pshed_lower_level[end]), fair_func)
    # # Plot total load shed over iterations
    # plot(1:iterations, pshed_lower_level, title = "Lower Level Load Shed over Iterations ($fair_func)", xlabel = "Iteration", ylabel = "Total Load Shed (kW)", marker = :o)
    # savefig("$save_path/lower_level_load_shed_over_iterations_$fair_func.svg")
    # Save load shed data to CSV
    df = DataFrame(Iteration = 1:iterations, Total_Load_Shed = pshed_lower_level, Lower_Level_Load_Shed = pshed_lower_level, Upper_Level_Load_Shed = pshed_upper_level)
    CSV.write("$save_path/load_shed_data_$fair_func.csv", df)
    # Display the plot with all three lines
    #display(plot(1:iterations, [pshed_lower_level pshed_upper_level], labels = ["Lower-Level Load Shed" "Upper-Level Load Shed"], title = "Load Shed over Iterations ($fair_func)", xlabel = "Iteration", ylabel = "Load Shed (kW)", marker = :o))
    savefig("$save_path/load_shed_comparison_over_iterations_$fair_func.svg")
    # Plot the load shed comparison, y-axis upper level load shed, x-axis lower level load shed
    # color each iteration differently
    iter_annotation = []
    for i in 1:iterations
        push!(iter_annotation, string(i))
    end
    pshed_comparison = scatter(pshed_lower_level, pshed_upper_level, title = "Load Shed Comparison ($fair_func)", xlabel = "Lower-Level Load Shed (kW)", ylabel = "Upper-Level Load Shed (kW)", marker = :o, label = "Iteration")
    for i in 1:iterations
        annotate!(pshed_comparison, pshed_lower_level[i], pshed_upper_level[i], text(string(i), :left, :green, 30))
    end
    # add a 45 degree line to the pshed_comparison plot 
    min_val = minimum([minimum(pshed_lower_level), minimum(pshed_upper_level)])
    max_val = maximum([maximum(pshed_lower_level), maximum(pshed_upper_level)])
    x_vals = range(min_val, max_val, length=100)
    y_vals = x_vals
    plot!(pshed_comparison, x_vals, y_vals, label = "y=x", line = (:dash, :red))
    #display(pshed_comparison)
    savefig("$save_path/load_shed_comparison_$fair_func.svg")

    plot!(relaxed_fldp_pshed_plot, pshed_lower_level, label = fair_func, xlabel = "Iteration", ylabel = "Lower-Level Load Shed (kW)", title = "Relaxed PADP Lower Level Load Shed Comparison")
    plot!(relaxed_fldp_weight_plot, weight_vals, label = fair_func, xlabel = "Load Index", ylabel = "Final Weight Value", title = "Relaxed PADP Final Weights Comparison")
    
    # Calculate fairness metric value for final load shed
    fairness_metric_value_prop = sum(log.(1 .+ (pshed_val .+ 1e-6)))
    fairness_metric_value_min_max = (maximum(pshed_val) - minimum(pshed_val))
    fairness_metric_value_palma = palma_ratio(pshed_val)
    fairness_metric_value_jain = jains_index(pshed_val)
    fair_metric_value_gini = gini_index(pshed_val)
    push!(relaxed_fldp_fairness_table, (fair_func, fairness_metric_value_min_max, fairness_metric_value_prop, fairness_metric_value_jain, fairness_metric_value_palma, fair_metric_value_gini))

    casepath = "results/$(Dates.today())/$case/"
    savefig(relaxed_fldp_pshed_plot, joinpath(casepath,"relaxed_padp_pshed_plot.svg"))
    savefig(relaxed_fldp_weight_plot, joinpath(casepath,"relaxed_padp_weight_plot.svg"))
    CSV.write(joinpath(casepath,"relaxed_padp_fairness_table.csv"),relaxed_fldp_fairness_table)

    # Find the switch samples from the relaxed solution using solve_mc_mld_shed_implicit_diff
    mld_implicit_diff_relaxed = solve_mc_mld_shed_implicit_diff(math_relaxed, ipopt; ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])
    imp_diff_model = instantiate_mc_model(
        math_relaxed,
        LinDist3FlowPowerModel,
        build_mc_mld_shedding_implicit_diff;
        ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]
    )
    ref = imp_diff_model.ref[:it][:pmd][:nw][0]
    # Check the voltage source consistency in the ref dictionary
    voltage_source_check(math, ref) 

    switch_states_extracted, block_status_extracted = extract_switch_block_states(mld_implicit_diff_relaxed["solution"])

    # Determine the number of rounding rounds and bernoulli samples per round
    n_rounds = 5 # Change the random seed per round to get different results
    n_bernoulli_samples = 10

    bernoulli_switch_selection_exp = Vector{Dict{Int, Float64}}()
    bernoulli_block_selection_exp = Vector{Dict{Int, Float64}}()
    bernoulli_load_selection_exp = Vector{Dict{Int, Float64}}()

    bernoulli_selection_index = []
    bernoulli_samples = Dict{Int,Vector{Dict{Int, Float64}}}()
    for r in 1:n_rounds
        rng = 100*r
        #rng = Random.MersenneTwister(100 * r)
        # Generate bernoulli samples for switches and blocks
        bernoulli_samples[r] = generate_bernoulli_samples(switch_states_extracted, n_bernoulli_samples, rng)

        # Find the best bernoulli sample be topology feasible and closes to the relaxed solution
        index, switch_states_radial, block_ids, block_status_radial, load_ids, load_status = radiality_check(ref, switch_states_extracted, block_status_extracted, bernoulli_samples[r])
        @info "Round $r: Best radial sample index: $index"
        @info "Round $r: Best radial sample switch status: $switch_states_radial"
        @info "Round $r: Best radial sample block status: $block_status_radial"
        @info "Round $r: Best radial sample load status: $load_status"
        push!(bernoulli_selection_index, index)
        push!(bernoulli_switch_selection_exp, switch_states_radial)
        push!(bernoulli_block_selection_exp, zip(block_ids, block_status_radial) |> Dict)
        push!(bernoulli_load_selection_exp, zip(load_ids, load_status) |> Dict)
    end

    # Create R copies of the math dictionary for random rounding testing
    math_random_test = Vector{Dict{String, Any}}()
    for r in 1:n_rounds
        math_copy = deepcopy(math_relaxed)
        push!(math_random_test, math_copy)
    end
    math_out = Vector{Dict{String, Any}}()
    # Apply the best switch configuration from each round to the respective math dictionary
    for r in 1:n_rounds
        push!(math_out, update_network(math_random_test[r], bernoulli_block_selection_exp[r], bernoulli_load_selection_exp[r], bernoulli_switch_selection_exp[r], ref, r))
        voltage_source_check(math, math_out[r])
    end

    # Test the AC feasibility of each rounded solution
    # Use the PMD IVRUPowerModel for AC power flow testing
    ac_feas = Vector{Dict{String, Any}}()
    # Allowing the solution to be reached iteration limit
    for r in 1:n_rounds
        @info "Testing AC feasibility for rounded solution from round $r"
        # Set the sourcebus generation capacity to a high value to simulate the slack bus(connection to transmission system)
        ls_percent = 1000
        for (i,gen) in math_out[r]["gen"]
            if gen["source_id"] == "voltage_source.source"
                pd_phase1=0
                pd_phase2=0
                pd_phase3=0
                qd_phase1=0
                qd_phase2=0
                qd_phase3=0
                for (ind, d) in math_out[r]["load"]
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

        feas_dict = ac_feasibility_test(math_out[r], r)
        push!(ac_feas, feas_dict)
    end

    # Count voltage violations from AC feasibility results
    bus_con_num, overvoltage_violation_num, undervoltage_violation_num, v_viol_over, v_viol_under = count_voltage_violations(math_out, ac_feas)
    println("Total bus connections checked: $bus_con_num")
    println("Total overvoltage violations: $overvoltage_violation_num")
    println("Total undervoltage violations: $undervoltage_violation_num")
    #p = PowerModelsDistribution.calc_admittance_matrix( math_relaxed)

    best_feasibility, max_load_served = export_ac_feasibility_results(math_out, ac_feas, save_path)

    best_set, best_mld = find_best_mld_solution(math_out, ipopt)
    @info "Best MLD solution found from set $best_set with objective value: $(best_mld["objective"])"
end
# # plot the best solution 
# eng_out = PowerModelsDistribution.transform_data_model(math_out[best_feasibility["set_id"]])

# p = powerplot(eng_out, bus = (:data=>"bus_type", :data_type=>"nominal"),
#                     branch = (:data=>"index", :data_type=>"ordinal"),
#                     gen    = (:data=>"pmax", :data_type=>"quantitative"),
#                     load   = (:data=>"pd",  :data_type=>"quantitative"),
#                     shunt = (:data=>"gs", :data_type=>"quantitative"),
#                     title = "Best AC Feasible Solution from Random Rounding",
#                     width = 300, height=300
# )
