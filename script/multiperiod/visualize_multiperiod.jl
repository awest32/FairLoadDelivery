"""
    visualize_multiperiod.jl

    Standalone visualization script for multiperiod FALD results.
    Loads saved JLD2 data from compare_fair_funcs_multiperiod.jl and generates heatmaps.

    Usage:
        julia --project=. script/multiperiod/visualize_multiperiod.jl [results_dir]

    If no results_dir is given, uses the most recent results directory.
"""

using FairLoadDelivery
using PowerModelsDistribution
using Plots
using Statistics
using JLD2
using Dates
using Colors

const PMD = PowerModelsDistribution

# ============================================================
# LOCATE RESULTS
# ============================================================
if length(ARGS) >= 1
    save_dir = ARGS[1]
else
    # Find most recent results directory
    base = "results"
    dates = filter(d -> isdir(joinpath(base, d, "bilevel_comparisons_multiperiod")), readdir(base))
    if isempty(dates)
        error("No multiperiod results found in results/")
    end
    latest = sort(dates)[end]
    save_dir = joinpath(base, latest, "bilevel_comparisons_multiperiod")
end

jld2_path = joinpath(save_dir, "multiperiod_results.jld2")
if !isfile(jld2_path)
    error("Results file not found: $jld2_path\nRun compare_fair_funcs_multiperiod.jl first.")
end

println("Loading results from: $jld2_path")
saved = JLD2.load(jld2_path)
results = saved["results"]
final_weights_results = saved["final_weights_results"]
rounding_results = saved["rounding_results"]
failed_combinations = saved["failed_combinations"]
CASES = saved["CASES"]
FAIR_FUNCS = saved["FAIR_FUNCS"]
N_PERIODS = saved["N_PERIODS"]
LOAD_SCALE_FACTORS = saved["LOAD_SCALE_FACTORS"]
LS_PERCENT = saved["LS_PERCENT"]
SOURCE_PU = saved["SOURCE_PU"]

println("  Cases: $CASES")
println("  Fair funcs: $FAIR_FUNCS")
println("  Periods: $N_PERIODS, scales: $LOAD_SCALE_FACTORS")

# ============================================================
# HEATMAP VISUALIZATIONS
# ============================================================
println("\n" * "=" ^ 70)
println("GENERATING MULTIPERIOD HEATMAP VISUALIZATIONS")
println("=" ^ 70)

critical_buses = []  # Match computation script

for case in CASES
    # Reconstruct network for bus/load name mapping
    _, math_viz, _, _ = FairLoadDelivery.setup_network("ieee_13_aw_edit/$case.dss", LS_PERCENT, SOURCE_PU, critical_buses)

    bus_name_map = build_bus_name_maps(math_viz)
    nw_ids_sorted = [string(i) for i in 0:(N_PERIODS-1)]
    n_periods = N_PERIODS
    period_labels = ["T$(i+1)\n(×$(LOAD_SCALE_FACTORS[i+1]))" for i in 0:(N_PERIODS-1)]

    # ── Compute bus distances from source via BFS ──
    # Identify virtual switch buses
    virtual_buses = Set{Int}()
    for (bid_str, bus) in math_viz["bus"]
        if contains(get(bus, "name", ""), "_virtual")
            push!(virtual_buses, parse(Int, bid_str))
        end
    end
    # Build adjacency from branches + switches
    adj = Dict{Int, Vector{Int}}()
    for (_, br) in math_viz["branch"]
        f, t = br["f_bus"], br["t_bus"]
        if !haskey(adj, f); adj[f] = Int[]; end
        if !haskey(adj, t); adj[t] = Int[]; end
        push!(adj[f], t)
        push!(adj[t], f)
    end
    if haskey(math_viz, "switch")
        for (_, sw) in math_viz["switch"]
            f, t = sw["f_bus"], sw["t_bus"]
            if !haskey(adj, f); adj[f] = Int[]; end
            if !haskey(adj, t); adj[t] = Int[]; end
            push!(adj[f], t)
            push!(adj[t], f)
        end
    end
    # Find source bus
    source_bus = nothing
    for (_, gen) in math_viz["gen"]
        if gen["source_id"] == "voltage_source.source"
            source_bus = gen["gen_bus"]
            break
        end
    end
    if source_bus === nothing
        source_bus = first(keys(adj))  # fallback
    end
    # BFS — virtual buses cost 0 distance
    bus_dist = Dict{Int, Int}()
    queue = [(source_bus, 0)]
    bus_dist[source_bus] = 0
    while !isempty(queue)
        bus, d = popfirst!(queue)
        for nb in get(adj, bus, Int[])
            if !haskey(bus_dist, nb)
                nd = nb in virtual_buses ? d : d + 1
                bus_dist[nb] = nd
                push!(queue, (nb, nd))
            end
        end
    end

    # ── Order loads by (bus distance from source, load name) ──
    all_load_ids = sort(parse.(Int, collect(keys(math_viz["load"]))))
    load_order = sort(all_load_ids, by=lid -> begin
        load = math_viz["load"][string(lid)]
        d = get(bus_dist, load["load_bus"], 999)
        name = get(load, "name", "zzz_$lid")
        (d, name)
    end)
    sorted_load_ids = load_order
    load_names = [get(math_viz["load"][string(lid)], "name", "load_$lid") for lid in sorted_load_ids]

    # Build multinetwork for original demand reference
    mn_data_viz = Dict{String,Any}("nw" => Dict{String,Any}())
    for t in 1:N_PERIODS
        nw_data = deepcopy(math_viz)
        scale = LOAD_SCALE_FACTORS[t]
        for (_, load) in nw_data["load"]
            load["pd"] = load["pd"] .* scale
            load["qd"] = load["qd"] .* scale
        end
        mn_data_viz["nw"][string(t-1)] = nw_data
    end

    # ── Order buses by distance from source (for load shed / voltage plots) ──
    bus_loads = Dict{Int, Vector{Int}}()
    for lid in sorted_load_ids
        bus_id = math_viz["load"][string(lid)]["load_bus"]
        if !haskey(bus_loads, bus_id)
            bus_loads[bus_id] = Int[]
        end
        push!(bus_loads[bus_id], lid)
    end
    bus_ids_sorted = sort(collect(keys(bus_loads)), by=bid -> get(bus_dist, bid, 999))
    bus_labels = [get(bus_name_map, bid, "bus_$bid") for bid in bus_ids_sorted]
    # Filter out virtual switch buses
    real_bus_ids = filter(bid -> !contains(get(math_viz["bus"][string(bid)], "name", ""), "_virtual"),
        parse.(Int, collect(keys(math_viz["bus"]))))
    all_bus_ids = sort(real_bus_ids, by=bid -> get(bus_dist, bid, 999))
    all_bus_labels = [get(bus_name_map, bid, "bus_$bid") for bid in all_bus_ids]

    # ── 1. WEIGHTS: multi-pane heatmap ──
    if haskey(final_weights_results, case)
        println("\n  Creating weight heatmaps for $case...")
        weight_panes = Plots.Plot[]
        any_critical = false
        valid_funcs = [ff for ff in FAIR_FUNCS if haskey(final_weights_results[case], ff)]
        for fair_func in FAIR_FUNCS
            is_first = !isempty(valid_funcs) && fair_func == first(valid_funcs)
            is_last = !isempty(valid_funcs) && fair_func == last(valid_funcs)
            if !haskey(final_weights_results[case], fair_func)
                push!(weight_panes, plot(title=FAIR_FUNC_LABELS[fair_func], grid=false, axis=false))
                continue
            end
            wdata = final_weights_results[case][fair_func]
            wid_to_val = Dict(zip(wdata[:weight_ids], wdata[:weights]))

            n_loads = length(sorted_load_ids)
            weight_matrix = zeros(n_periods, n_loads)
            for (j, lid) in enumerate(sorted_load_ids)
                w = get(wid_to_val, lid, NaN)
                for i in 1:n_periods
                    weight_matrix[i, j] = w
                end
            end

            # Clamp weights >10 for color scale; mark critical loads separately
            crit_x = Float64[]
            crit_y = Float64[]
            for j in 1:length(sorted_load_ids), i in 1:n_periods
                if weight_matrix[i, j] > 20.0
                    push!(crit_x, Float64(j))
                    push!(crit_y, Float64(i))
                    weight_matrix[i, j] = 10.0  # clamp for heatmap color
                end
            end

            p = heatmap(load_names, period_labels, weight_matrix,
                title=FAIR_FUNC_LABELS[fair_func],
                color=:RdYlBu, xrotation=45,
                clims=(1, 10), colorbar=false,
                ylabel=is_first ? "Period" : "",
                ytickfontcolor=is_first ? :black : RGBA(0,0,0,0),
                titlefontsize=10, tickfontsize=7)
            # Overlay star markers on critical load cells (weight > 10)
            if !isempty(crit_x)
                any_critical = true
                scatter!(p, crit_x, crit_y,
                    marker=:star5, markersize=8, markercolor=:black,
                    markerstrokecolor=:white, markerstrokewidth=1.5,
                    label=false)
            end
            # Light grid lines at cell boundaries
            n_loads = length(sorted_load_ids)
            vline!(p, (0:n_loads) .+ 0.5, color=:gray70, linewidth=0.5, label=false)
            hline!(p, (0:n_periods) .+ 0.5, color=:gray70, linewidth=0.5, label=false)
            push!(weight_panes, p)
        end
        n_panes = length(weight_panes)
        println("    Panes: $n_panes")
        # Horizontal colorbar as gradient image on bottom row
        cb_data = reshape(range(1, 10, length=256), 1, :)
        p_cb = heatmap(range(1, 10, length=256), [""], cb_data,
            color=:RdYlBu, clims=(1, 10), colorbar=false,
            yticks=false, xticks=1:1:10,
            xlabel=any_critical ? "Weight  (★ = critical load, w > 20)" : "Weight",
            tickfontsize=8,
            framestyle=:box, top_margin=-5Plots.mm)
        lay = @layout [grid(1, n_panes){0.93h}; a{0.07h}]
        p_combined = plot(weight_panes..., p_cb,
            layout=lay,
            size=(400 * n_panes, 650), bottom_margin=10Plots.mm,
            plot_title="Final Weights (Period × Load): $case",
            plot_titlefontsize=12)
        savefig(p_combined, joinpath(save_dir, "heatmap_weights_all_$case.svg"))
        println("    Saved: heatmap_weights_all_$case.svg")
    end

    # ── 2. LOAD SHED: multi-pane heatmap ──
    if haskey(rounding_results, case)
        println("\n  Creating load shed heatmaps for $case...")
        shed_panes = Plots.Plot[]
        valid_funcs = [ff for ff in FAIR_FUNCS if haskey(rounding_results[case], ff)]
        for fair_func in FAIR_FUNCS
            is_first = !isempty(valid_funcs) && fair_func == first(valid_funcs)
            is_last = !isempty(valid_funcs) && fair_func == last(valid_funcs)
            if !haskey(rounding_results[case], fair_func)
                push!(shed_panes, plot(title=FAIR_FUNC_LABELS[fair_func], grid=false, axis=false))
                continue
            end
            per_period = rounding_results[case][fair_func]

            n_buses = length(bus_ids_sorted)
            shed_matrix = fill(NaN, n_periods, n_buses)

            for (i, nw_id) in enumerate(nw_ids_sorted)
                if !haskey(per_period, nw_id) || per_period[nw_id]["best_mld"] === nothing || isempty(per_period[nw_id]["best_mld"])
                    continue
                end
                mld_sol = per_period[nw_id]["best_mld"]["solution"]["load"]
                orig_nw = mn_data_viz["nw"][nw_id]
                for (j, bus_id) in enumerate(bus_ids_sorted)
                    pd_total = 0.0
                    pshed_total = 0.0
                    for lid in bus_loads[bus_id]
                        lid_str = string(lid)
                        pd_total += sum(orig_nw["load"][lid_str]["pd"])
                        if haskey(mld_sol, lid_str) && haskey(mld_sol[lid_str], "pshed")
                            pshed_total += sum(mld_sol[lid_str]["pshed"])
                        end
                    end
                    shed_matrix[i, j] = pd_total > 0 ? (pshed_total / pd_total) * 100 : 0.0
                end
            end

            replace!(shed_matrix, NaN => 0.0)
            p = heatmap(bus_labels, period_labels, shed_matrix,
                title=FAIR_FUNC_LABELS[fair_func],
                color=:RdYlBu, xrotation=45,
                clims=(0, 100), colorbar=false,
                ylabel=is_first ? "Period" : "",
                ytickfontcolor=is_first ? :black : RGBA(0,0,0,0),
                titlefontsize=10, tickfontsize=7,
)
            # Light grid lines at cell boundaries
            n_buses_shed = length(bus_ids_sorted)
            vline!(p, (0:n_buses_shed) .+ 0.5, color=:gray70, linewidth=0.5, label=false)
            hline!(p, (0:n_periods) .+ 0.5, color=:gray70, linewidth=0.5, label=false)
            push!(shed_panes, p)
        end
        n_panes = length(shed_panes)
        println("    Panes: $n_panes")
        cb_data = reshape(range(0, 100, length=256), 1, :)
        p_cb = heatmap(range(0, 100, length=256), [""], cb_data,
            color=:RdYlBu, clims=(0, 100), colorbar=false,
            yticks=false, xlabel="Shed (%)", tickfontsize=8,
            framestyle=:box, top_margin=-5Plots.mm)
        lay = @layout [grid(1, n_panes){0.93h}; a{0.07h}]
        p_combined = plot(shed_panes..., p_cb,
            layout=lay,
            size=(400 * n_panes, 650), bottom_margin=10Plots.mm,
            plot_title="Load Shed % (Period × Bus): $case",
            plot_titlefontsize=12)
        savefig(p_combined, joinpath(save_dir, "heatmap_loadshed_all_$case.svg"))
        println("    Saved: heatmap_loadshed_all_$case.svg")

        # ── 3. VOLTAGE: multi-pane heatmap ──
        println("\n  Creating voltage heatmaps for $case...")
        volt_panes = Plots.Plot[]
        valid_funcs = [ff for ff in FAIR_FUNCS if haskey(rounding_results[case], ff)]
        for fair_func in FAIR_FUNCS
            is_first = !isempty(valid_funcs) && fair_func == first(valid_funcs)
            is_last = !isempty(valid_funcs) && fair_func == last(valid_funcs)
            if !haskey(rounding_results[case], fair_func)
                push!(volt_panes, plot(title=FAIR_FUNC_LABELS[fair_func], grid=false, axis=false))
                continue
            end
            per_period = rounding_results[case][fair_func]

            n_buses = length(all_bus_ids)
            voltage_matrix = fill(NaN, n_periods, n_buses)

            for (i, nw_id) in enumerate(nw_ids_sorted)
                if !haskey(per_period, nw_id)
                    continue
                end
                period_res = per_period[nw_id]
                ac_feas_vec = period_res["ac_feas"]
                if isempty(ac_feas_vec)
                    continue
                end
                ac_sol = nothing
                for feas in ac_feas_vec
                    if feas["feas_status"]
                        ac_sol = feas["solution"]
                        break
                    end
                end
                if ac_sol === nothing
                    ac_sol = ac_feas_vec[1]["solution"]
                end

                for (j, bus_id) in enumerate(all_bus_ids)
                    bid_str = string(bus_id)
                    if !haskey(ac_sol, "bus") || !haskey(ac_sol["bus"], bid_str)
                        continue
                    end
                    bs = ac_sol["bus"][bid_str]
                    if haskey(bs, "vr") && haskey(bs, "vi")
                        voltage_matrix[i, j] = mean(sqrt.(bs["vr"].^2 .+ bs["vi"].^2))
                    elseif haskey(bs, "vm")
                        voltage_matrix[i, j] = mean(bs["vm"])
                    end
                end
            end

            # Track de-energized cells (NaN or near-zero voltage)
            # Place markers at grid intersections (bottom-left corner of de-energized cell)
            deenerg_x = Float64[]
            deenerg_y = Float64[]
            for i in 1:n_periods, j in 1:length(all_bus_ids)
                if isnan(voltage_matrix[i, j]) || voltage_matrix[i, j] < 0.01
                    push!(deenerg_x, Float64(j) - 0.5)
                    push!(deenerg_y, Float64(i) - 0.5)
                end
            end
            replace!(voltage_matrix, NaN => 0.0)
            p = heatmap(all_bus_labels, period_labels, voltage_matrix,
                title=FAIR_FUNC_LABELS[fair_func],
                color=:RdYlBu, xrotation=45,
                clims=(0.9, 1.1), colorbar=false,
                ylabel=is_first ? "Period" : "",
                ytickfontcolor=is_first ? :black : RGBA(0,0,0,0),
                titlefontsize=10, tickfontsize=7)
            # Overlay × on de-energized buses
            if !isempty(deenerg_x)
                scatter!(p, deenerg_x, deenerg_y,
                    marker=:xcross, markersize=5, markercolor=:white,
                    markerstrokecolor=:black, markerstrokewidth=1.5,
                    label=false)
            end
            # Light grid lines at cell boundaries
            n_buses_volt = length(all_bus_ids)
            vline!(p, (0:n_buses_volt) .+ 0.5, color=:gray70, linewidth=0.5, label=false)
            hline!(p, (0:n_periods) .+ 0.5, color=:gray70, linewidth=0.5, label=false)
            push!(volt_panes, p)
        end
        n_panes = length(volt_panes)
        println("    Panes: $n_panes")
        cb_data = reshape(range(0.9, 1.1, length=256), 1, :)
        p_cb = heatmap(range(0.9, 1.1, length=256), [""], cb_data,
            color=:RdYlBu, clims=(0.9, 1.1), colorbar=false,
            yticks=false, xlabel="Voltage (p.u.)  (× = de-energized)", tickfontsize=8,
            framestyle=:box, top_margin=-5Plots.mm)
        lay = @layout [grid(1, n_panes){0.93h}; a{0.07h}]
        p_combined = plot(volt_panes..., p_cb,
            layout=lay,
            size=(400 * n_panes, 650), bottom_margin=10Plots.mm,
            plot_title="Bus Voltage p.u. (Period × Bus): $case",
            plot_titlefontsize=12)
        savefig(p_combined, joinpath(save_dir, "heatmap_voltage_all_$case.svg"))
        println("    Saved: heatmap_voltage_all_$case.svg")
    end
end

println("\nAll multiperiod heatmap visualizations complete.")
