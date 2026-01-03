# Create the other fairness functions 
using FairLoadDelivery
using JuMP, Ipopt, Gurobi

# Function to compute Jain's Fairness Index
function jains_fairness_index(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64})
    model = JuMP.Model(Ipopt.Optimizer)
    n = length(pshed_prev)
    @variable(model, pshed_new[1:n] >= 0)
    @variable(model, weights_new[1:n] >= 0)
    @constraint(model, [i=1:n], weights_new[i] <= 10)
    @constraint(model, [i=1:length(weights_prev)], weights_new[i]-weights_prev[i]<= 0.1)
    @constraint(model, [i in 1:n],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:n)
    )
    sum_pshed = sum(pshed_new)
    sum_pshed_squared = sum(pshed_new[i]^2 for i in 1:n)
    if sum_pshed == 0.0
        return 0.0
    end
    
    fairness_index = (sum_pshed^2) / (n * sum_pshed_squared)
    @objective(model, Max, fairness_index)
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

# Function to compute the min max of load shed 
# pshed is a vector of load shed values
# updating pshed with the gradietn dpshed_dw
# with respect the the change in weights w, w_prev
# optimizing pshed_new and weights_new
function min_max_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64})
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, pshed_new[1:length(pshed_prev)] >= 0)
    @variable(model, weights_new[1:length(weights_prev)] >= 0)
    @constraint(model, [i=1:length(weights_prev)], weights_new[i] <= 10)
    @constraint(model, [i=1:length(weights_prev)], weights_new[i]-weights_prev[i]<= 0.1)

    #@variable(model, t >= 0)
    @constraint(model, [i in 1:length(pshed_prev)],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:length(weights_prev))
    )
    #@constraint(model, t >= maximum(pshed_new))
    @objective(model, Min, maximum(pshed_new))
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

# Function to compute the proportional fairness load shed
function proportional_fairness_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64})
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, pshed_new[1:length(pshed_prev)] >= 0)
    @variable(model, weights_new[1:length(weights_prev)] >= 0)
    @constraint(model, weights_new[1:length(weights_prev)] .<= 10)
    @constraint(model, [i=1:length(weights_prev)], weights_new[i]-weights_prev[i]<= 0.1)
    @constraint(model, [i in 1:length(pshed_prev)],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:length(weights_prev))
    )
    @objective(model, Max, sum(log(pshed_new[i] + 1e-6) for i in 1:length(pshed_new))) # Adding a small constant to avoid log(0)
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

# Function to compute complete efficiency (alpha fairness) of load shed
function complete_efficiency_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64})
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, pshed_new[1:length(pshed_prev)] >= 0)
    @variable(model, weights_new[1:length(weights_prev)] >= 0)
    @constraint(model, weights_new[1:length(weights_prev)] .<= 10)
    @constraint(model, [i=1:length(weights_prev)], weights_new[i]-weights_prev[i]<= 0.1)
    @constraint(model, [i in 1:length(pshed_prev)],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:length(weights_prev))
    )
    @objective(model, Min, sum(pshed_new[i] for i in 1:length(pshed_new)))
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

# Function to compute the infinity norm fairness of load shed
function infinity_norm_fairness_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64})
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, pshed_new[1:length(pshed_prev)] >= 0)
    @variable(model, weights_new[1:length(weights_prev)] >= 0)
    @variable(model, t >= 0)
    @constraint(model, [i in 1:length(pshed_prev)],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:length(weights_prev))
    )
    @constraint(model, [i in 1:length(pshed_new)],
        pshed_new[i] <= t
    )
    @objective(model, Min, t)
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

# Define the fairness functions for post processing
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


# Plot the fairness indices for the final load shed values
using Plots
function plot_fairness_indices(res, pshed::Vector{Float64}, weight_ids::Vector{Int}, iteration::Int, exp_folder::String, test_name::String)
    n = length(pshed)
    sum_pshed = sum(pshed)
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
    load_shed = []
    idxs = sort(parse.(Int,collect(keys(res["load"]))))
    for i in 1:length(idxs)
        load = res["load"][string(i)]
        for idx in 1:length(load["pd"])
            push!(load_served, load["pd"][idx])
        end
        # push!(load_shed, load["pshed"])
    end
    load_served_sum = sum(load_served)

    switch_statuses = Dict{String, Any}()
    for (id, switch) in res["switch"]
        switch_statuses[id] = switch["state"]
    end

    switch_names = Dict{String, Any}()
    for (id, switch) in ref[:switch]
        @info id
        @info switch
        switch_names[string(id)] = switch["name"]
    end
    println("Load served percentage: $(load_served_sum/load_ref_sum*100) %")
    push!(served, (load_served_sum/load_ref_sum)*100)

    # Calculate and print fairness indices
    served_array = collect(load_served./load_ref)
    df_fairness_results = DataFrame(TestName=String[], GiniIndex=Float64[], JainsIndex=Float64[], PalmaRatio=Float64[], AlphaFairness1=Float64[])

    push!(df_fairness_results.TestName, "$test_name")
    push!(df_fairness_results.GiniIndex, gini_index(served_array))
    push!(df_fairness_results.JainsIndex, jains_index(served_array))
    push!(df_fairness_results.PalmaRatio, palma_ratio(served_array))
    push!(df_fairness_results.AlphaFairness1, alpha_fairness(served_array, 1))
    println(df_fairness_results)
    CSV.write("$exp_folder/fairness_indices_iteration_$iteration.csv", df_fairness_results)
    # Plot the fairness indices
    indices = ["Gini Index", "Jain's Index", "Palma Ratio",
                "Alpha Fairness (α=1)"]
    values = [df_fairness_results.GiniIndex[1], df_fairness_results.JainsIndex[1],
              df_fairness_results.PalmaRatio[1], df_fairness_results.AlphaFairness1[1]]
    bar_plot = bar(indices, values, title = "Fairness Indices at Iteration $iteration",
                   ylabel = "Value", legend = false, ylim = (0, maximum(values)*1.2))
    savefig(bar_plot, "$exp_folder/fairness_indices_iteration_$iteration.svg")
    display(bar_plot)
end