using FairLoadDelivery

# Create a folder for the fairness experiment results
today = Dates.today()
fairness_exp_folder = "results/$(today)/$CASE/bilevel_exp/$fair_func"
if !isdir(fairness_exp_folder)
    mkdir(fairness_exp_folder)
end
best_solution = Dict{String, Any}()
objectives = zeros(Float64,length(ac_results))
iterations = zeros(Int, length(ac_results))
for i in 1:length(ac_results)
    if !isempty(ac_results[i])
        pf_ac = ac_results[i]
        objective_value = pf_ac["objective"]
        @info "Round $i AC power flow objective value: $objective_value"
        objectives[i] = objective_value
        iterations[i] = i
        @warn "Round $i AC power flow did not solve successfully. Termination status: $(pf_ac["termination_status"])"
    end
end
max_index = argmax(objectives);
best_network = math_out[max_index];

best_solution = mld_results[max_index];
plot_network_load_shed(best_solution["solution"], best_network;
    output_file=joinpath(fairness_exp_folder, "FALD_Solution.svg"),
    layout=:ieee13)


