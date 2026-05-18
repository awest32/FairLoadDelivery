"""
    Results / plotting block for run_validation_mn.jl.

    Re-runs only Step 5 (heatmap + final-result summary) and the report
    generation, using variables already in `Main` from a completed
    run_validation_mn.jl execution:
      mn_rounded_solutions, mn_new, mn_relaxed_final, nw_ids_sorted,
      N_PERIODS, LOAD_SCALE_FACTORS, PEAK_TIME_COSTS,
      CASE, FAIR_FUNC, pshed_type, save_dir, validation_results.

    Usage (in REPL after a full run):
        include("script/bilevel_validation/results_block_mn.jl")
"""

using StatsPlots
using FairLoadDelivery

# Unified 9pt font defaults for every figure produced here.
include(joinpath(@__DIR__, "../figure_defaults.jl"))

# Representative periods for grouped bar (override before include() to customize).
# Default picks 3 evenly-spaced indices into 1:N_PERIODS so it adapts to
# downsampled runs (T=8 → [1, 4, 8]) as well as the full T=24 day.
if !@isdefined(REP_PERIODS)
    REP_PERIODS = N_PERIODS <= 3 ?
        collect(1:N_PERIODS) :
        unique([1, max(1, N_PERIODS ÷ 2), N_PERIODS])
end

print_validation_header("Step 5: Load-shed heatmap + final result")

ref_load_ids = sort(collect(keys(mn_new["nw"][nw_ids_sorted[1]]["load"])), by=x->parse(Int, x))
load_labels  = [mn_new["nw"][nw_ids_sorted[1]]["load"][lid]["name"] for lid in ref_load_ids]
period_labels = ["t=$t" for t in 1:N_PERIODS]

pshed_matrix = fill(NaN, N_PERIODS, length(ref_load_ids))
for (t, nw_id) in enumerate(nw_ids_sorted)
    haskey(mn_rounded_solutions, nw_id) || continue
    sol_t = mn_rounded_solutions[nw_id]["solution"]
    haskey(sol_t, "load") || continue
    for (j, lid) in enumerate(ref_load_ids)
        if haskey(sol_t["load"], lid) && haskey(sol_t["load"][lid], "pshed")
            pshed_matrix[t, j] = sum(sol_t["load"][lid]["pshed"])
        end
    end
end

# ---- Bus on/off status across load buses × ALL periods. Integer MLD fully
# sheds or serves each load, so served_fraction lives in {0, 1} per
# (period, bus). Restricted to buses that actually host a load.
math_ref = mn_new["nw"][nw_ids_sorted[1]]
bus_name_map = FairLoadDelivery.build_bus_name_maps(math_ref)
load_bus_set = Set(math_ref["load"][lid]["load_bus"] for lid in ref_load_ids)
all_bus_ids = sort(collect(load_bus_set))
bus_labels = [get(bus_name_map, bid, "bus_$bid") for bid in all_bus_ids]

# pshed_matrix column j → load_id, so map each load to its bus column index
bus_col = Dict(bid => k for (k, bid) in enumerate(all_bus_ids))
load_to_bus_col = [bus_col[math_ref["load"][lid]["load_bus"]] for lid in ref_load_ids]

pd_ref_matrix    = zeros(N_PERIODS, length(ref_load_ids))
bus_pshed_matrix = zeros(N_PERIODS, length(all_bus_ids))
bus_pd_matrix    = zeros(N_PERIODS, length(all_bus_ids))
for (t, nw_id) in enumerate(nw_ids_sorted)
    nw_data = mn_new["nw"][nw_id]
    for (j, lid) in enumerate(ref_load_ids)
        pd_total = sum(nw_data["load"][lid]["pd"])
        pd_ref_matrix[t, j] = pd_total
        bus_pd_matrix[t, load_to_bus_col[j]] += pd_total
        v = pshed_matrix[t, j]
        bus_pshed_matrix[t, load_to_bus_col[j]] += isnan(v) ? 0.0 : v
    end
end

# Bus-level z_demand (= served indicator): 1.0 = served, 0.0 = shed.
# The block constraint forces all loads at a bus to share shed status, so
# bus_status_matrix takes binary values when the rounded MLD solution
# respects integrality. Equivalent to z_demand for any load at the bus.
bus_status_matrix = fill(NaN, N_PERIODS, length(all_bus_ids))
for t in 1:N_PERIODS, b in 1:length(all_bus_ids)
    if bus_pd_matrix[t, b] > 1e-9
        bus_status_matrix[t, b] = 1.0 - bus_pshed_matrix[t, b] / bus_pd_matrix[t, b]
    end
end

p_heat = heatmap(bus_labels, period_labels, bus_status_matrix,
    xlabel = "Bus",
    ylabel = "Period",
    color  = cgrad(["#E5EFEA", "#2A6F6B"]),  # pale sage (shed) → muted teal (served)
    clims  = (0.0, 1.0),
    xrotation = 45,
    yticks = (1:N_PERIODS, period_labels),
    colorbar = false,
)
display(p_heat)
savefig(p_heat, joinpath(save_dir, "loadshed_heatmap_$(pshed_type)_$case.svg"))

# ---- Grouped bar over representative periods (matches min_max_trade_off_mn style) ----
rep_valid = filter(t -> 1 <= t <= N_PERIODS, REP_PERIODS)
if !isempty(rep_valid)
    # Build (N_loads × |rep|) matrix of pshed values for the grouped bar
    rep_matrix = zeros(length(ref_load_ids), length(rep_valid))
    for (k, t) in enumerate(rep_valid)
        for j in 1:length(ref_load_ids)
            v = pshed_matrix[t, j]
            rep_matrix[j, k] = isnan(v) ? 0.0 : v
        end
    end
    rep_labels = reshape(["t=$t" for t in rep_valid], 1, length(rep_valid))
    p_grouped = groupedbar(load_labels, rep_matrix;
        bar_position = :dodge,
        labels = rep_labels,
        xlabel = "load",
        ylabel = "load shed (kW)",
        title  = "$FAIR_FUNC / $pshed_type — rep. periods",
        legend = :topright,
        linecolor = :black,
    )
    display(p_grouped)
    savefig(p_grouped, joinpath(save_dir, "loadshed_grouped_$(pshed_type)_$case.svg"))
end

valid_mask = .!isnan.(pshed_matrix)
period_total = [sum(pshed_matrix[t, j] for j in 1:length(ref_load_ids) if valid_mask[t, j]; init=0.0) for t in 1:N_PERIODS]
period_max   = [maximum(pshed_matrix[t, j] for j in 1:length(ref_load_ids) if valid_mask[t, j]; init=0.0) for t in 1:N_PERIODS]

total_shed_all = sum(filter(!isnan, pshed_matrix))
finite_vals    = filter(!isnan, vec(pshed_matrix))
global_max     = isempty(finite_vals) ? 0.0 : maximum(finite_vals)
max_t, max_j   = isempty(finite_vals) ? (0, 0) : Tuple(argmax(replace(pshed_matrix, NaN => -Inf)))
max_load_label = (max_j == 0) ? "n/a" : load_labels[max_j]

rounded_objectives = Dict{Int,Float64}()
for (t, nw_id) in enumerate(nw_ids_sorted)
    if haskey(mn_rounded_solutions, nw_id) && haskey(mn_rounded_solutions[nw_id], "objective")
        rounded_objectives[t] = mn_rounded_solutions[nw_id]["objective"]
    end
end

println("\n  Final result (rounded, raw — not cost-weighted):")
println("    Total shed (Σ over all loads × periods) = $(round(total_shed_all, digits=3)) kW")
println("    Global max-shed point: load=$max_load_label, period=t=$max_t, value=$(round(global_max, digits=3)) kW")
println("    Relaxed multi-period MLD objective       = $(round(mn_relaxed_final["objective"], digits=3))")
println("\n  Per-period summary:")
println("    " * rpad("t", 4) * rpad("scale", 8) * rpad("λ", 8) *
                  rpad("total shed", 14) * rpad("max shed", 12) * "rounded obj")
for t in 1:N_PERIODS
    obj_str = haskey(rounded_objectives, t) ? string(round(rounded_objectives[t], digits=3)) : "—"
    println("    " * rpad(string(t), 4) *
                    rpad(string(LOAD_SCALE_FACTORS[t]), 8) *
                    rpad(string(PEAK_TIME_COSTS[t]), 8) *
                    rpad(string(round(period_total[t], digits=3)), 14) *
                    rpad(string(round(period_max[t], digits=3)), 12) *
                    obj_str)
end

validation_results["final"] = Dict(
    "total_shed_all_periods" => total_shed_all,
    "global_max_shed"        => global_max,
    "global_max_load"        => max_load_label,
    "global_max_period"      => max_t,
    "period_total_shed"      => period_total,
    "period_max_shed"        => period_max,
    "rounded_objectives"     => rounded_objectives,
    "relaxed_mn_objective"   => mn_relaxed_final["objective"],
)

report_path = joinpath(save_dir, "validation_report_mn_$(pshed_type)_$case.txt")
generate_summary_report(validation_results, report_path)
println("\nResults block complete. Heatmap → $(joinpath(save_dir, "loadshed_heatmap_$(pshed_type)_$case.svg"))")
println("Report → $report_path")
