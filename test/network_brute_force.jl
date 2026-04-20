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

# ============================================================
# CONFIGURATION
# ============================================================
const CASE = "ieee_13_aw_edit/motivation_c.dss"
const LS_PERCENT = 0.8  # 10% load shed, 90% generation capacity

# Save results
save_dir = "results/$(Dates.today())/brute_force_motivation_c"
mkpath(save_dir)

# Solver
gurobi_solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

# Visualization constants
const FAIR_FUNC_COLORS = Dict(
    "jain"       => RGB(0.2, 0.6, 0.2),   # green
    "gini"       => RGB(0.8, 0.4, 0.0),   # orange
    "palma"      => RGB(0.6, 0.2, 0.6),   # purple
    "efficiency" => RGB(0.2, 0.4, 0.8),   # blue
)

const FAIR_FUNC_LABELS = Dict(
    "jain"       => "Jain's Index",
    "gini"       => "Gini Coefficient",
    "palma"      => "Palma Ratio",
    "efficiency" => "Efficiency",
)

# ============================================================
# HELPER FUNCTIONS (adapted from compare_fair_funcs_and_networks.jl)
# ============================================================

function extract_load_shed(mld::Dict)
    if mld === nothing || !haskey(mld, "solution") || !haskey(mld["solution"], "load")
        @error "MLD solution does not contain load data"
    end

    total_pshed = 0.0
    total_pd_served = 0.0
    for (load_id, load_data) in mld["solution"]["load"]
        if haskey(load_data, "pshed")
            total_pshed += sum(load_data["pshed"])
        end
        if haskey(load_data, "pd")
            total_pd_served += sum(load_data["pd"])
        end
    end
    return total_pshed, total_pd_served
end

function extract_per_load_data(mld::Dict, math::Dict; original_math::Union{Dict,Nothing}=nothing)
    orig_math = something(original_math, math)

    # Group loads by bus
    bus_loads = Dict{Int, Vector{Int}}()
    for (lid_str, load_data) in math["load"]
        lid = parse(Int, lid_str)
        bus_id = load_data["load_bus"]
        if !haskey(bus_loads, bus_id)
            bus_loads[bus_id] = Int[]
        end
        push!(bus_loads[bus_id], lid)
    end

    bus_ids = sort(collect(keys(bus_loads)))
    pshed_pct = Float64[]
    pd_served_pct = Float64[]
    qshed_pct = Float64[]
    qd_served_pct = Float64[]
    pd_served_kw = Float64[]
    pd_demand_kw = Float64[]

    sol_load = get(get(mld, "solution", Dict()), "load", Dict{String,Any}())

    for bus_id in bus_ids
        pd_total = 0.0
        qd_total = 0.0
        pd_served = 0.0
        qd_served = 0.0

        for lid in bus_loads[bus_id]
            lid_str = string(lid)
            orig_load = orig_math["load"][lid_str]
            pd_total += sum(orig_load["pd"])
            qd_total += sum(orig_load["qd"])

            if haskey(sol_load, lid_str)
                load_sol = sol_load[lid_str]
                if haskey(load_sol, "pd")
                    pd_s = load_sol["pd"]
                    pd_served += isa(pd_s, AbstractArray) ? sum(pd_s) : pd_s
                end
                if haskey(load_sol, "qd")
                    qd_s = load_sol["qd"]
                    qd_served += isa(qd_s, AbstractArray) ? sum(qd_s) : qd_s
                end
            end
        end

        pshed = pd_total - pd_served
        qshed = qd_total - qd_served

        push!(pshed_pct, pd_total > 0 ? (pshed / pd_total) * 100 : 0.0)
        push!(pd_served_pct, pd_total > 0 ? (pd_served / pd_total) * 100 : 0.0)
        push!(qshed_pct, qd_total > 0 ? (qshed / qd_total) * 100 : 0.0)
        push!(qd_served_pct, qd_total > 0 ? (qd_served / qd_total) * 100 : 0.0)
        push!(pd_served_kw, pd_served)
        push!(pd_demand_kw, pd_total)
    end

    return bus_ids, pshed_pct, pd_served_pct, qshed_pct, qd_served_pct, pd_served_kw, pd_demand_kw
end

# ============================================================
# SETUP
# ============================================================

function setup_brute_force()
    @info "Setting up network for brute force enumeration..."
    eng, math, lbs, critical_id = setup_network(CASE, LS_PERCENT, [])

    # Build ref dictionary for update_network
    imp_model = instantiate_mc_model(math, LinDist3FlowPowerModel,
        build_mc_mld_shedding_implicit_diff;
        ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
    ref = imp_model.ref[:it][:pmd][:nw][0]

    # Log switch info
    @info "Switches in network:"
    switch_ids = sort(parse.(Int, collect(keys(math["switch"]))))
    for sw_id in switch_ids
        sw = math["switch"][string(sw_id)]
        @info "  Switch $sw_id: name=$(sw["name"]), state=$(sw["state"]), status=$(sw["status"])"
    end

    # Set initial switch states: close 632633, 632645, 670671; open all others
    initial_closed = Set(["632633", "632645", "670671"])
    switch_selection = Dict{Int, Float64}()
    for sw_id in switch_ids
        sw_name = math["switch"][string(sw_id)]["name"]
        switch_selection[sw_id] = sw_name in initial_closed ? 1.0 : 0.0
    end

    math = update_network(math, switch_selection, ref)
    @info "Initial switch configuration applied."

    return eng, math, lbs, critical_id, ref, switch_ids
end

# ============================================================
# ENUMERATE SWITCH CONFIGURATIONS
# ============================================================

function enumerate_switch_configs(switch_ids::Vector{Int})
    n_switches = length(switch_ids)
    n_configs = 2^n_switches
    configs = Vector{Dict{Int, Float64}}(undef, n_configs)

    for i in 0:(n_configs - 1)
        bits = digits(i, base=2, pad=n_switches)
        configs[i + 1] = Dict(sw_id => Float64(bit) for (sw_id, bit) in zip(switch_ids, bits))
    end

    return configs
end

# ============================================================
# RUN BRUTE FORCE
# ============================================================

function run_brute_force(math::Dict, ref::Dict, switch_ids::Vector{Int}, configs::Vector{Dict{Int, Float64}})
    n_configs = length(configs)
    bus_name_map = build_bus_name_maps(math)

    # Build switch name lookup
    switch_name_map = Dict{Int, String}()
    for sw_id in switch_ids
        switch_name_map[sw_id] = math["switch"][string(sw_id)]["name"]
    end

    # Results storage
    results = DataFrame(
        config_id = Int[],
        switch_config = String[],
        feasible = Bool[],
        termination_status = String[],
        total_pshed = Float64[],
        total_pd_served = Float64[],
        pct_served = Float64[],
        pct_shed = Float64[],
        jains = Float64[],
        gini = Float64[],
        palma = Float64[],
        total_load_served_kw = Float64[]
    )

    # Add switch name columns
    for sw_id in switch_ids
        results[!, Symbol("sw_$(switch_name_map[sw_id])")] = Float64[]
    end

    # Store per-load data for each feasible config
    per_load_results = Dict{Int, NamedTuple}()
    solutions = Dict{Int, Dict}()
    math_copies = Dict{Int, Dict}()

    @info "Running brute force: $n_configs configurations to solve..."

    for (cfg_idx, config) in enumerate(configs)
        config_str = join([Int(config[sw_id]) for sw_id in switch_ids], "")
        @info "Config $cfg_idx/$n_configs: switches=$config_str"

        # Apply switch configuration
        math_copy = update_network(math, config, ref)

        # Keep status=1 and dispatchable=1 for all switches so that
        # ref_add_load_blocks! includes them in block_switches.
        for (sw_id_str, sw_data) in math_copy["switch"]
            sw_data["status"] = 1.0
        end

        # Solve using the random rounding integer formulation, which has
        # constraint_set_switch_state_rounded to fix switch states from data
        # AND constraint_radial_topology to enforce tree structure.
        # (build_mc_mld_switchable_integer does NOT fix switch states.)
        local mld
        try
            mld = FairLoadDelivery.solve_mc_mld_shed_random_round_integer(math_copy, gurobi_solver)
        catch e
            @warn "Config $cfg_idx: solver error: $e"
            row = [cfg_idx, config_str, false, "ERROR", NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
            for sw_id in switch_ids
                push!(row, config[sw_id])
            end
            push!(results, row)
            continue
        end

        term_status = string(mld["termination_status"])
        feasible = term_status in ["LOCALLY_SOLVED", "OPTIMAL"]

        if !feasible
            @warn "Config $cfg_idx: infeasible ($term_status)"
            row = [cfg_idx, config_str, false, term_status, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
            for sw_id in switch_ids
                push!(row, config[sw_id])
            end
            push!(results, row)
            continue
        end

        # Extract results
        total_pshed, total_pd_served = extract_load_shed(mld)
        bus_ids, pshed_pct, pd_served_pct, qshed_pct, qd_served_pct, pd_served_kw, pd_demand_kw = extract_per_load_data(mld, math_copy)

        # Compute total demand and percent served/shed
        total_demand_kw = sum(pd_demand_kw)
        total_kw = sum(pd_served_kw)
        if config_str == "111111"
            @info "=== POST-SOLVE DEBUG for 111111 ==="
            @info "termination_status = $term_status"
            @info "solution keys: $(keys(mld["solution"]))"
            if haskey(mld["solution"], "switch")
                for (s, sw) in mld["solution"]["switch"]
                    @info "  switch $s: $(sw)"
                end
                break
            end
        end
        pct_served = total_demand_kw > 0 ? (total_pd_served / total_demand_kw) * 100 : 0.0
        pct_shed = total_demand_kw > 0 ? (total_pshed / total_demand_kw) * 100 : 0.0

        # Compute fairness metrics on percent served per bus
        jains_val = NaN
        gini_val = NaN
        palma_val = NaN

        try
            if all(x -> x > 0, pd_served_pct)
                jains_val = FairLoadDelivery.jains_index(pd_served_pct)
                gini_val = FairLoadDelivery.gini_index(pd_served_pct)
                palma_val = FairLoadDelivery.palma_ratio(pd_served_pct)
            elseif any(x -> x > 0, pd_served_pct)
                jains_val = FairLoadDelivery.jains_index(pd_served_pct)
                if sum(pd_served_pct) > 0
                    gini_val = FairLoadDelivery.gini_index(pd_served_pct)
                end
                sorted_pct = sort(pd_served_pct)
                n = length(sorted_pct)
                bottom_40 = sum(sorted_pct[1:floor(Int, 0.4n)])
                if bottom_40 > 0
                    palma_val = FairLoadDelivery.palma_ratio(pd_served_pct)
                else
                    top_10 = sum(sorted_pct[ceil(Int, 0.9n):end])
                    palma_val = top_10 > 0 ? Inf : NaN
                end
            end
        catch e
            @warn "Config $cfg_idx: fairness metric error: $e"
        end

        @info "  Feasible: pshed=$total_pshed ($(round(pct_shed, digits=2))%), served=$total_pd_served ($(round(pct_served, digits=2))%), Jain=$(round(jains_val, digits=4)), Gini=$(round(gini_val, digits=4)), Palma=$(round(palma_val, digits=4))"

        row = [cfg_idx, config_str, true, term_status, total_pshed, total_pd_served, pct_served, pct_shed, jains_val, gini_val, palma_val, total_kw]
        for sw_id in switch_ids
            push!(row, config[sw_id])
        end
        push!(results, row)

        # Store detailed results for later visualization
        per_load_results[cfg_idx] = (
            bus_ids = bus_ids,
            pshed_pct = pshed_pct,
            pd_served_pct = pd_served_pct,
            pd_served_kw = pd_served_kw,
            pd_demand_kw = pd_demand_kw
        )
        solutions[cfg_idx] = mld
        math_copies[cfg_idx] = math_copy
    end

    return results, per_load_results, solutions, math_copies
end

# ============================================================
# FIND BEST TOPOLOGIES
# ============================================================

function find_best_topologies(results::DataFrame, switch_ids::Vector{Int}, math::Dict)
    feasible = filter(row -> row.feasible, results)

    if nrow(feasible) == 0
        @error "No feasible configurations found!"
        return Dict{String, Vector{NamedTuple}}()
    end

    switch_name_map = Dict{Int, String}()
    for sw_id in switch_ids
        switch_name_map[sw_id] = math["switch"][string(sw_id)]["name"]
    end

    # Returns all tied topologies as a vector of NamedTuples
    winners = Dict{String, Vector{NamedTuple}}()

    # Helper: get switch state string for a config
    function switch_states_str(cfg_id)
        sw_states = []
        for sw_id in switch_ids
            col = Symbol("sw_$(switch_name_map[sw_id])")
            state = results[results.config_id .== cfg_id, col][1]
            push!(sw_states, "$(switch_name_map[sw_id])=$(Int(state))")
        end
        return join(sw_states, ", ")
    end

    function make_winner(row)
        return (
            config_id = row.config_id,
            switch_config = row.switch_config,
            metric_value_jains = row.jains,
            metric_value_gini = row.gini,
            metric_value_palma = row.palma,
            total_served = row.total_load_served_kw,
            pct_served = row.pct_served,
            pct_shed = row.pct_shed
        )
    end

    # --- Best Jain (highest) ---
    jain_col = feasible[!, :jains]
    jain_valid = .!isnan.(jain_col)
    if any(jain_valid)
        valid = feasible[jain_valid, :]
        best_val = maximum(valid[!, :jains])
        tied = valid[valid[!, :jains] .== best_val, :]
        tied = sort(tied, :total_load_served_kw, rev=true)
        winners["jain"] = [make_winner(tied[i, :]) for i in 1:nrow(tied)]
        @info "Best Jain's Index: $(best_val) — $(nrow(tied)) tied topologies"
        for i in 1:nrow(tied)
            @info "  $(tied[i, :switch_config]) | served=$(round(tied[i, :total_load_served_kw], digits=4)) | $(switch_states_str(tied[i, :config_id]))"
        end
    end

    # --- Best Gini (lowest) ---
    gini_col = feasible[!, :gini]
    gini_valid = .!isnan.(gini_col)
    if any(gini_valid)
        valid = feasible[gini_valid, :]
        best_val = minimum(valid[!, :gini])
        tied = valid[valid[!, :gini] .== best_val, :]
        tied = sort(tied, :total_load_served_kw, rev=true)
        winners["gini"] = [make_winner(tied[i, :]) for i in 1:nrow(tied)]
        @info "Best Gini Index: $(best_val) — $(nrow(tied)) tied topologies"
        for i in 1:nrow(tied)
            @info "  $(tied[i, :switch_config]) | served=$(round(tied[i, :total_load_served_kw], digits=4)) | $(switch_states_str(tied[i, :config_id]))"
        end
    end

    # --- Best Palma (closest to 1.0) ---
    try
        palma_col = feasible[!, :palma]
        palma_valid = .!isnan.(palma_col)
        n_valid = sum(palma_valid)
        @info "Valid Palma configs: $n_valid of $(nrow(feasible)) feasible"
        if n_valid > 0
            valid = feasible[palma_valid, :]
            # Filter out Inf — no meaningful Palma when bottom 40% is zero
            finite_mask = isfinite.(valid[!, :palma])
            if any(finite_mask)
                valid = valid[finite_mask, :]
                dists = abs.(valid[!, :palma] .- 1.0)
                best_dist = minimum(dists)
                tied = valid[dists .== best_dist, :]
                tied = sort(tied, :total_load_served_kw, rev=true)
                winners["palma"] = [make_winner(tied[i, :]) for i in 1:nrow(tied)]
                @info "Best Palma Ratio (closest to 1.0): dist=$(best_dist), value=$(tied[1, :palma]) — $(nrow(tied)) tied topologies"
                for i in 1:nrow(tied)
                    @info "  $(tied[i, :switch_config]) | palma=$(round(tied[i, :palma], digits=6)) | served=$(round(tied[i, :total_load_served_kw], digits=4)) | $(switch_states_str(tied[i, :config_id]))"
                end
            else
                @info "No finite Palma values — all configs have zero in bottom 40%"
            end
        end
    catch e
        @warn "Palma selection failed: $e"
    end

    # --- Most Efficient (highest total load served) ---
    best_served = maximum(feasible[!, :total_load_served_kw])
    tied = feasible[feasible[!, :total_load_served_kw] .== best_served, :]
    tied = sort(tied, :total_load_served_kw, rev=true)
    winners["efficiency"] = [make_winner(tied[i, :]) for i in 1:nrow(tied)]
    @info "Most Efficient: $(best_served) — $(nrow(tied)) tied topologies"
    for i in 1:nrow(tied)
        @info "  $(tied[i, :switch_config]) | served=$(round(tied[i, :total_load_served_kw], digits=4)) | $(switch_states_str(tied[i, :config_id]))"
    end

    return winners
end

# ============================================================
# GENERATE VISUALIZATIONS
# ============================================================

function generate_visualizations(
    results::DataFrame,
    winners::Dict{String, Vector{NamedTuple}},
    per_load_results::Dict{Int, NamedTuple},
    solutions::Dict{Int, Dict},
    math_copies::Dict{Int, Dict},
    math::Dict,
    switch_ids::Vector{Int},
    save_dir::String
)
    bus_name_map = build_bus_name_maps(math)
    switch_name_map = Dict{Int, String}()
    for sw_id in switch_ids
        switch_name_map[sw_id] = math["switch"][string(sw_id)]["name"]
    end

    # 1. Save summary CSV
    csv_path = joinpath(save_dir, "brute_force_results.csv")
    CSV.write(csv_path, results)
    @info "Results CSV saved to $csv_path"

    # 2. Network load shed plots for best winner per metric (first = most load served)
    for (metric, winner_list) in winners
        winner = winner_list[1]
        cfg_id = winner.config_id
        if !haskey(solutions, cfg_id) || !haskey(math_copies, cfg_id)
            @warn "No solution data for $metric winner (config $cfg_id)"
            continue
        end

        mld = solutions[cfg_id]
        math_copy = math_copies[cfg_id]

        output_file = joinpath(save_dir, "network_loadshed_best_$(metric).svg")
        try
            plot_network_load_shed(mld["solution"], math_copy;
                output_file=output_file, layout=:ieee123, width=48, height=36)
            @info "Network plot saved: $output_file"
        catch e
            @warn "Failed to generate network plot for $metric: $e"
        end
    end

    # 3. Per-bus bar chart for best winner per metric
    for (metric, winner_list) in winners
        winner = winner_list[1]
        cfg_id = winner.config_id
        if !haskey(per_load_results, cfg_id)
            continue
        end

        data = per_load_results[cfg_id]
        bus_names = [get(bus_name_map, bid, string(bid)) for bid in data.bus_ids]

        p = bar(
            bus_names, data.pshed_pct,
            xlabel = "Bus",
            ylabel = "Load Shed (%)",
            title = "Load Shed per Bus — Best $(uppercase(metric)) ($(winner.switch_config))",
            legend = false,
            xrotation = 45,
            size = (900, 500),
            left_margin = 10Plots.mm,
            bottom_margin = 15Plots.mm,
            color = get(FAIR_FUNC_COLORS, metric, RGB(0.4, 0.4, 0.8)),
            bar_width = 0.6,
            ylims = (0, 100)
        )

        savefig(p, joinpath(save_dir, "per_bus_loadshed_best_$(metric).svg"))
        @info "Per-bus bar chart saved for $metric"
    end

    # 3b. Summary of all tied best topologies per metric
    tied_rows = DataFrame(metric=String[], switch_config=String[], jains=Float64[], gini=Float64[], palma=Float64[], total_served_kw=Float64[], pct_served=Float64[], pct_shed=Float64[])
    for metric in sort(collect(keys(winners)))
        for w in winners[metric]
            push!(tied_rows, (metric, w.switch_config, w.metric_value_jains, w.metric_value_gini, w.metric_value_palma, w.total_served, w.pct_served, w.pct_shed))
        end
    end
    tied_path = joinpath(save_dir, "best_topologies_all_tied.csv")
    CSV.write(tied_path, tied_rows)
    @info "Tied best topologies saved to $tied_path"

    # 4. Summary comparison plot: first winner per metric side-by-side
    active_metrics = sort(collect(keys(winners)))
    if !isempty(active_metrics)
        # Collect union of all bus names (using first winner per metric)
        all_bus_ids = Set{Int}()
        for metric in active_metrics
            cfg_id = winners[metric][1].config_id
            if haskey(per_load_results, cfg_id)
                union!(all_bus_ids, per_load_results[cfg_id].bus_ids)
            end
        end
        sorted_bus_ids = sort(collect(all_bus_ids))
        bus_names = [get(bus_name_map, bid, string(bid)) for bid in sorted_bus_ids]
        n_buses = length(sorted_bus_ids)

        n_metrics = length(active_metrics)
        group_width = 0.8
        bar_w = group_width / n_metrics
        x_positions = collect(1:n_buses)

        p = Plots.plot(
            xlabel = "Bus",
            ylabel = "Load Shed (%)",
            title = "Load Shed Comparison: Best Topologies per Metric",
            legend = :topright,
            xticks = (x_positions, bus_names),
            xrotation = 45,
            size = (1400, 600),
            left_margin = 10Plots.mm,
            bottom_margin = 15Plots.mm,
            top_margin = 5Plots.mm,
            right_margin = 20Plots.mm,
            ylims = (0, 100)
        )

        for (mi, metric) in enumerate(active_metrics)
            cfg_id = winners[metric][1].config_id
            if !haskey(per_load_results, cfg_id)
                continue
            end

            data = per_load_results[cfg_id]
            bid_to_pshed = Dict(zip(data.bus_ids, data.pshed_pct))

            vals = [get(bid_to_pshed, bid, 0.0) for bid in sorted_bus_ids]
            offsets = x_positions .+ (mi - (n_metrics + 1) / 2) * bar_w

            color = get(FAIR_FUNC_COLORS, metric, RGB(0.5, 0.5, 0.5))
            label_str = get(FAIR_FUNC_LABELS, metric, titlecase(metric))

            bar!(p, offsets, vals,
                bar_width = bar_w,
                label = "$label_str ($(winners[metric][1].switch_config))",
                color = color,
                linecolor = :black,
                linewidth = 0.5
            )
        end

        savefig(p, joinpath(save_dir, "comparison_winners_loadshed.svg"))
        @info "Comparison plot saved."
    end

    # 5. Summary statistics
    feasible_results = filter(row -> row.feasible, results)
    @info "=== BRUTE FORCE SUMMARY ==="
    @info "Total configurations: $(nrow(results))"
    @info "Feasible configurations: $(nrow(feasible_results))"
    @info "Infeasible configurations: $(nrow(results) - nrow(feasible_results))"

    if nrow(feasible_results) > 0
        @info "Load served range: $(round(minimum(feasible_results.total_load_served_kw), digits=4)) — $(round(maximum(feasible_results.total_load_served_kw), digits=4)) kW"
        @info "Percent served range: $(round(minimum(feasible_results.pct_served), digits=2))% — $(round(maximum(feasible_results.pct_served), digits=2))%"
        @info "Percent shed range: $(round(minimum(feasible_results.pct_shed), digits=2))% — $(round(maximum(feasible_results.pct_shed), digits=2))%"

        valid_jain = filter(row -> !isnan(row.jains), feasible_results)
        if nrow(valid_jain) > 0
            @info "Jain's Index range: $(round(minimum(valid_jain.jains), digits=4)) — $(round(maximum(valid_jain.jains), digits=4))"
        end

        valid_gini = filter(row -> !isnan(row.gini), feasible_results)
        if nrow(valid_gini) > 0
            @info "Gini Index range: $(round(minimum(valid_gini.gini), digits=4)) — $(round(maximum(valid_gini.gini), digits=4))"
        end

        palma_col = feasible_results[!, :palma]
        palma_valid = palma_col[.!isnan.(palma_col)]
        if length(palma_valid) > 0
            @info "Palma Ratio range: $(round(minimum(palma_valid), digits=4)) — $(round(maximum(palma_valid), digits=4))"
        end
    end
end

# ============================================================
# MAIN EXECUTION
# ============================================================

function main()
    @info "=== Brute Force Network Testing: motivation_c ==="
    @info "Date: $(Dates.today())"

    # Setup
    eng, math, lbs, critical_id, ref, switch_ids = setup_brute_force()

    # Enumerate all switch configurations
    configs = enumerate_switch_configs(switch_ids)
    @info "Generated $(length(configs)) switch configurations for $(length(switch_ids)) switches"

    # Run brute force
    results, per_load_results, solutions, math_copies = run_brute_force(math, ref, switch_ids, configs)

    # Find best topologies
    winners = find_best_topologies(results, switch_ids, math)

    # Generate visualizations and save results
    generate_visualizations(results, winners, per_load_results, solutions, math_copies, math, switch_ids, save_dir)

    @info "=== Brute force complete. Results saved to $save_dir ==="

    return results, winners, per_load_results, solutions, math_copies
end

# Run
results, winners, per_load_results, solutions, math_copies = main()
nothing
