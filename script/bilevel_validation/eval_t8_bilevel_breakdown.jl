#=
    Per-load / per-period / per-bus breakdown of the T=8 bilevel solutions.

    Loads the three T=8 bilevel JLD2s (palma, gini, min_max) and emits a clean
    three-way comparison of WHERE and WHEN load shed lands, for both the integer
    (rounded `pshed_matrix`) and relaxed (`relaxed_pshed_matrix`) solutions.

    Outputs CSVs under results/<today>/t8_bilevel_eval/ and prints summaries:
      * t8_per_load_<variant>.csv    rows = 9 loads,    cols = palma|gini|min_max (+ pd_total)
      * t8_per_period_<variant>.csv  rows = 8 periods,  cols = palma|gini|min_max (+ hour, lambda)
      * t8_per_bus_<variant>.csv     rows = load buses, cols = palma|gini|min_max

    Usage: julia --project=. script/bilevel_validation/eval_t8_bilevel_breakdown.jl
=#
using JLD2, DataFrames, CSV, Printf, Dates

const CASE       = "case6_unbalanced_switch_more_meshed_good4integer"
const FAIR_FUNCS = ["palma", "gini", "min_max"]
const ROOT       = joinpath(@__DIR__, "../../results/2026-06-04/bilevel_validation_mn", CASE)
const OUT        = joinpath(@__DIR__, "../../results", Dates.format(now(), "yyyy-mm-dd"), "t8_bilevel_eval")
mkpath(OUT)

_load(ff) = JLD2.load(joinpath(ROOT, "$(ff)_absolute",
                               "bilevel_mn_$(CASE)_$(ff)_absolute.jld2"))
D = Dict(ff => _load(ff) for ff in FAIR_FUNCS)

ref         = D["palma"]
load_labels = ref["load_labels"]
bus_labels  = ref["bus_labels"]
hours       = ref["SELECTED_HOURS"]
lambda      = ref["PEAK_TIME_COSTS"]
nP          = length(hours)

# Sanity: label order must match across fair funcs for the columns to align.
for ff in FAIR_FUNCS
    @assert D[ff]["load_labels"] == load_labels "load_labels mismatch for $ff"
    @assert D[ff]["bus_labels"]  == bus_labels  "bus_labels mismatch for $ff"
end

for (vtag, mkey, buskey) in [("integer", "pshed_matrix", "bus_pshed_matrix"),
                             ("relaxed", "relaxed_pshed_matrix", "relaxed_bus_pshed_matrix")]
    # --- per-load: Σ over periods (kW) ---
    perload = DataFrame(load = load_labels)
    for ff in FAIR_FUNCS
        perload[!, ff] = round.(vec(sum(D[ff][mkey], dims = 1)), digits = 2)
    end
    perload[!, "pd_total"] = round.(vec(sum(ref["pd_ref_matrix"], dims = 1)), digits = 2)
    CSV.write(joinpath(OUT, "t8_per_load_$(vtag).csv"), perload)

    # --- per-period: Σ over loads (kW) ---
    perper = DataFrame(period = 1:nP, hour = hours, lambda = lambda)
    for ff in FAIR_FUNCS
        perper[!, ff] = round.(vec(sum(D[ff][mkey], dims = 2)), digits = 2)
    end
    CSV.write(joinpath(OUT, "t8_per_period_$(vtag).csv"), perper)

    # --- per-bus: Σ over periods (kW) ---
    perbus = DataFrame(bus = bus_labels)
    for ff in FAIR_FUNCS
        perbus[!, ff] = round.(vec(sum(D[ff][buskey], dims = 1)), digits = 2)
    end
    CSV.write(joinpath(OUT, "t8_per_bus_$(vtag).csv"), perbus)

    # --- console summary ---
    println("\n==================== $(uppercase(vtag)) ====================")
    totals = [round(sum(D[ff][mkey]), digits = 1) for ff in FAIR_FUNCS]
    println("total shed (kW):  ", join(["$(ff)=$(t)" for (ff, t) in zip(FAIR_FUNCS, totals)], "  "))
    println("\nPER-LOAD (Σ_t kW):");   show(perload, allrows = true, allcols = true); println()
    println("\nPER-PERIOD (Σ_load kW):"); show(perper, allrows = true, allcols = true); println()
    println("\nPER-BUS (Σ_t kW):");    show(perbus, allrows = true, allcols = true); println()
end

println("\nCSVs written to: ", OUT)
