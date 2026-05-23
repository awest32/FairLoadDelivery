"""
    Standalone per-bus load-shed heatmap replot from a saved bilevel-mn JLD2.

    Renders the period × bus served-fraction heatmap (mirrors results_block_mn.jl).
    Bus aggregation is the right unit because it mimics how load blocks
    actually get switched on/off.

    Two paths:

    1. NEW JLD2 (post-2026-05-17 schema): bus_status_matrix + bus_labels are
       loaded directly. Fast — no FairLoadDelivery import needed.

    2. OLD JLD2: only pshed_matrix + load_labels + CASE are saved. Falls back
       to re-running setup_network + create_multinetwork_data_profiled to
       reconstruct the load→bus mapping and per-load pd reference, then
       rebuilds bus_status_matrix. Requires FairLoadDelivery and the OpenDSS
       case file. Adjust the BACKFILL_CONFIG block to match the original
       run's settings if they differ from run_validation_mn.jl defaults.

    Each saved run is keyed by (CASE, FAIR_FUNC, pshed_type) — JLD2 lives at
    results/<date>/bilevel_validation_mn/<CASE>/<FAIR_FUNC>_<pshed_type>/
        bilevel_mn_<CASE>_<FAIR_FUNC>_<pshed_type>.jld2.
"""

using JLD2
using Plots
using Dates

include(joinpath(@__DIR__, "../figure_defaults.jl"))

CASE       = "case6_unbalanced_switch_more_meshed_good4integer"
FAIR_FUNC  = "efficiency"
pshed_type = "absolute"

# ============================================================
# BACKFILL_CONFIG — only used if the JLD2 predates the bus-data schema.
# Match these to the run_validation_mn.jl settings used to produce the JLD2.
# ============================================================
BACKFILL_CONFIG = Dict(
    "case6_unbalanced_switch_more_meshed_good4integer" => (
        case_file        = joinpath(@__DIR__, "../../data/pmd_opendss/case6_unbalanced_switch_more_meshed_good4integer.dss"),
        ls_percent       = 0.8,
        switch_rating    = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)]) * 0.8,
        selected_hours   = [4, 6, 8, 12, 15, 18, 20, 22],
        peak_stress      = 1.0,
        center_at_nominal = true,
    ),
    "motivation_c_good4integer" => (
        case_file        = joinpath(@__DIR__, "../../data/ieee_13_aw_edit/motivation_c_good4integer.dss"),
        ls_percent       = 0.8,
        switch_rating    = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)]) * 0.8,
        selected_hours   = [4, 6, 8, 12, 15, 18, 20, 22],
        peak_stress      = 1.0,
        center_at_nominal = true,
    ),
)

function _find_latest_jld2(case::String, fair_func::String, pshed_type::String)
    base = joinpath(@__DIR__, "../../results")
    isdir(base) || error("results dir not found: $base")
    dates = sort(filter(d -> isdir(joinpath(base, d, "bilevel_validation_mn", case,
                                            "$(fair_func)_$(pshed_type)")),
                        readdir(base)); rev=true)
    isempty(dates) && error("No saved run found for case=$case, fair_func=$fair_func, pshed_type=$pshed_type")
    return joinpath(base, dates[1], "bilevel_validation_mn", case,
                    "$(fair_func)_$(pshed_type)",
                    "bilevel_mn_$(case)_$(fair_func)_$(pshed_type).jld2")
end

jld_path = _find_latest_jld2(CASE, FAIR_FUNC, pshed_type)
isfile(jld_path) || error("JLD2 file not found: $jld_path")

println("Loading bilevel run data → $jld_path")
saved = JLD2.load(jld_path)

CASE       = saved["CASE"]
FAIR_FUNC  = saved["FAIR_FUNC"]
pshed_type = saved["pshed_type"]
N_PERIODS  = saved["N_PERIODS"]
save_dir   = dirname(jld_path)

if haskey(saved, "bus_status_matrix") && haskey(saved, "bus_labels")
    println("Using saved bus-level matrices (new JLD2 schema).")
    bus_labels        = saved["bus_labels"]
    bus_status_matrix = saved["bus_status_matrix"]
else
    println("Old JLD2 schema — backfilling bus data via setup_network + create_multinetwork_data_profiled.")
    haskey(BACKFILL_CONFIG, CASE) ||
        error("No BACKFILL_CONFIG entry for case '$CASE'. Add one matching the original run's settings.")
    cfg = BACKFILL_CONFIG[CASE]
    isfile(cfg.case_file) || error("Backfill case file not found: $(cfg.case_file)")

    using FairLoadDelivery
    pshed_matrix = saved["pshed_matrix"]
    load_labels  = saved["load_labels"]

    eng, math, lbs, critical_id = FairLoadDelivery.setup_network(
        cfg.case_file, cfg.ls_percent; switch_rating = cfg.switch_rating)

    # Pick selected_hours that match the saved N_PERIODS. cfg.selected_hours is
    # an OK default but newer runs use T=24 (collect(0:23)). Fall back to a
    # sensible default by N_PERIODS so old + new JLD2s both backfill.
    selected_hours_eff = if length(cfg.selected_hours) == N_PERIODS
        cfg.selected_hours
    elseif N_PERIODS == 24
        collect(0:23)
    elseif N_PERIODS == 8
        [4, 6, 8, 12, 15, 18, 20, 22]
    elseif N_PERIODS == 3
        [6, 12, 18]
    else
        error("BACKFILL_CONFIG[$CASE].selected_hours has length $(length(cfg.selected_hours)) " *
              "but JLD2 reports N_PERIODS=$N_PERIODS. No automatic mapping available — " *
              "edit BACKFILL_CONFIG[$CASE].selected_hours to match the original run.")
    end
    if selected_hours_eff !== cfg.selected_hours
        @info "Backfill: using N_PERIODS-derived selected_hours=$selected_hours_eff " *
              "(BACKFILL_CONFIG had $(cfg.selected_hours))"
    end

    mn_data = FairLoadDelivery.create_multinetwork_data_profiled(
        math, N_PERIODS;
        hours = selected_hours_eff,
        peak_stress = cfg.peak_stress,
        center_at_nominal = cfg.center_at_nominal)

    nw_ids_sorted = sort(collect(keys(mn_data["nw"])), by = x -> parse(Int, x))
    math_ref = mn_data["nw"][nw_ids_sorted[1]]
    ref_load_ids = sort(collect(keys(math_ref["load"])), by = x -> parse(Int, x))

    if length(ref_load_ids) != length(load_labels)
        error("Backfill load count ($(length(ref_load_ids))) ≠ saved load_labels ($(length(load_labels))). " *
              "Config probably doesn't match the original run.")
    end

    backfill_load_names = [math_ref["load"][lid]["name"] for lid in ref_load_ids]
    if backfill_load_names != load_labels
        @warn "Backfill load NAME order does not match saved load_labels — pshed columns will be misaligned. " *
              "Saved: $(load_labels). Reconstructed: $(backfill_load_names)."
        error("Refusing to render misaligned heatmap. Check that the .dss file and config match the original run.")
    end

    bus_name_map = FairLoadDelivery.build_bus_name_maps(math_ref)
    load_bus_set = Set(math_ref["load"][lid]["load_bus"] for lid in ref_load_ids)
    all_bus_ids  = sort(collect(load_bus_set))
    bus_labels   = [get(bus_name_map, bid, "bus_$bid") for bid in all_bus_ids]
    bus_col      = Dict(bid => k for (k, bid) in enumerate(all_bus_ids))
    load_to_bus_col = [bus_col[math_ref["load"][lid]["load_bus"]] for lid in ref_load_ids]

    bus_pshed_matrix = zeros(N_PERIODS, length(all_bus_ids))
    bus_pd_matrix    = zeros(N_PERIODS, length(all_bus_ids))
    for (t, nw_id) in enumerate(nw_ids_sorted)
        nw_data = mn_data["nw"][nw_id]
        for (j, lid) in enumerate(ref_load_ids)
            pd_total = sum(nw_data["load"][lid]["pd"])
            bus_pd_matrix[t, load_to_bus_col[j]] += pd_total
            v = pshed_matrix[t, j]
            bus_pshed_matrix[t, load_to_bus_col[j]] += isnan(v) ? 0.0 : v
        end
    end

    bus_status_matrix = fill(NaN, N_PERIODS, length(all_bus_ids))
    for t in 1:N_PERIODS, b in 1:length(all_bus_ids)
        if bus_pd_matrix[t, b] > 1e-9
            bus_status_matrix[t, b] = 1.0 - bus_pshed_matrix[t, b] / bus_pd_matrix[t, b]
        end
    end
end

period_labels = ["t=$t" for t in 1:N_PERIODS]

# Force binary (served = 1, any shed = 0) so the heatmap renders strictly two
# colors. Buses with multiple loads where only some are shed otherwise show
# intermediate gradient shades; for the load-block representation, any shed
# on a bus = the block is off.
bus_status_binary = map(bus_status_matrix) do v
    isnan(v) && return NaN
    v >= 1.0 - 1e-9 ? 1.0 : 0.0
end

# Number buses 1..N in stable order and persist the mapping back to their
# original names so plots can be cross-referenced.
bus_index_labels = string.(1:length(bus_labels))
println("\nBus index → name mapping:")
for (i, name) in enumerate(bus_labels)
    println("  $i\t→\t$name")
end
map_path = joinpath(save_dir, "bus_index_map_$(pshed_type)_$(CASE).txt")
open(map_path, "w") do io
    println(io, "# Bus index → original bus name mapping for $(CASE) / $(FAIR_FUNC) / $(pshed_type)")
    println(io, "# index\tname")
    for (i, name) in enumerate(bus_labels)
        println(io, "$i\t$name")
    end
end
println("Bus index map → $map_path")

p_heat = heatmap(bus_index_labels, period_labels, bus_status_binary,
    xlabel = "Bus",
    ylabel = "Period",
    color  = cgrad(["#E5EFEA", "#2A6F6B"]),  # pale sage (shed) → muted teal (served)
    clims  = (0.0, 1.0),
    xrotation = 0,
    yticks = (1:N_PERIODS, period_labels),
    colorbar = false,
    size = (700, 500),
    left_margin = 5Plots.mm,
    right_margin = 5Plots.mm,
    bottom_margin = 3Plots.mm,
    top_margin = 6Plots.mm,
    tickfontsize = 12,
    guidefontsize = 12,
    titlefontsize = 12,
    legendfontsize = 12,
)
display(p_heat)
out_path = joinpath(save_dir, "loadshed_heatmap_replot_$(pshed_type)_$(CASE)_$(FAIR_FUNC).svg")
savefig(p_heat, out_path)
println("Per-bus heatmap → $out_path")
