"""
    Post-hoc fairness norms table across (problem, fair_func) pairs.

    For each pair, locates the latest JLD2 in `results/<date>/...`, computes
    the per-load aggregate-shed vector (sum over periods × phases — same
    convention as `min_max_trade_off_mn.jl` and `palma_trade_off_mn.jl`),
    and reports:

        L1   = Σ_i |shed_i|       (== total kW shed)
        L2   = sqrt(Σ_i shed_i^2)
        L∞   = max_i shed_i
        CoV  = std / mean  (NaN if mean ≈ 0)

    Six rows by default:
        single-level × {efficiency, min_max, palma}
        bilevel      × {efficiency, min_max, palma}

    The trade-off rows are evaluated at the α slice nearest
    `TRADE_OFF_ALPHA_TARGET` (default 0.75 — matches the heatmap convention
    in trade_off_heatmap_mn.jl). Efficiency JLD2s only contain α=0 samples,
    so the nearest-to-target lookup degenerates to that single slice; min_max
    and palma JLD2s span 0..1, so 0.75 picks a paper-relevant fairness
    weighting.

    Output:
      * Console table (pretty-printed markdown).
      * CSV at results/<today>/post_hoc_fairness/<case>_<pshed_type>_fairness_norms.csv.

    Usage:
        julia --project=. script/post_hoc_fairness_table.jl
        # or in the REPL:
        include("script/post_hoc_fairness_table.jl")

    Override CASE_KEY / PSHED_TYPE at the top before include.
"""

using JLD2
using LinearAlgebra
using Statistics
using Dates
using DataFrames
using CSV
using Printf

# Map a short case key to the filename tags used by the two pipelines (they
# disagree — bilevel uses the full DSS basename, trade-off scripts use a short
# tag). Extend when adding new cases.
const CASE_TAGS = Dict(
    "case6" => (bilevel  = "case6_unbalanced_switch_more_meshed_good4integer",
                trade_off = "more_meshed_6_bus"),
    "motivation_c" => (bilevel  = "motivation_c_good4integer",
                       trade_off = "13_bus"),
)

# ============================================================
# CONFIGURATION
# ============================================================
CASE_KEY   = "case6"
PSHED_TYPE = "absolute"
# α slice for trade-off rows. Same convention as trade_off_heatmap_mn.jl —
# 0.75 lets readers cross-reference the heatmap directly. Efficiency JLD2s
# only contain α=0 anyway, so the lookup is harmless there.
TRADE_OFF_ALPHA_TARGET = 0.75
# Skip rows for which no JLD2 is found (silent). When false, missing rows are
# reported with NaN entries so the user sees what's missing.
SKIP_MISSING = false

@assert haskey(CASE_TAGS, CASE_KEY) "Unknown CASE_KEY=$CASE_KEY (known: $(collect(keys(CASE_TAGS))))"
tags = CASE_TAGS[CASE_KEY]

# ============================================================
# JLD2 PATH RESOLVERS
# ============================================================
results_root = joinpath(@__DIR__, "../results")

function _latest_jld2(rel_path_fn)
    isdir(results_root) || error("results dir not found: $results_root")
    candidates = String[]
    for d in readdir(results_root)
        full = joinpath(results_root, d, rel_path_fn(d))
        isfile(full) && push!(candidates, full)
    end
    isempty(candidates) && return nothing
    return sort(candidates; rev = true)[1]   # lexicographic on dates = chronological
end

function _trade_off_jld2(fair_func::String)
    if fair_func in ("efficiency", "min_max")
        return _latest_jld2(_ -> joinpath("trade_off_mn",
            "$(fair_func)_trade_off_mn_$(tags.trade_off)_$(PSHED_TYPE).jld2"))
    elseif fair_func == "palma"
        return _latest_jld2(_ -> joinpath("palma_trade_off_mn",
            "palma_sweep_mn_$(tags.trade_off)_$(PSHED_TYPE).jld2"))
    end
    return nothing
end

function _bilevel_jld2(fair_func::String)
    return _latest_jld2(_ -> joinpath("bilevel_validation_mn", tags.bilevel,
        "$(fair_func)_$(PSHED_TYPE)",
        "bilevel_mn_$(tags.bilevel)_$(fair_func)_$(PSHED_TYPE).jld2"))
end

# ============================================================
# PER-LOAD AGGREGATE SHED EXTRACTORS
# ============================================================
function _trade_off_shed_vec(path::String, fair_func::String)
    saved = JLD2.load(path)
    per_load_agg = saved["per_load_agg"]    # alpha × load
    alphas       = saved["alphas"]
    idx = argmin(abs.(alphas .- TRADE_OFF_ALPHA_TARGET))
    return (vec = collect(per_load_agg[idx, :]),
            α   = alphas[idx],
            n_periods = saved["N_PERIODS"],
            source = path)
end

function _bilevel_shed_vec(path::String)
    saved = JLD2.load(path)
    pshed_matrix = saved["pshed_matrix"]    # period × load (NaN where unsolved)
    # Sum across periods, treating NaN as 0 (un-solved periods drop out).
    n_loads = size(pshed_matrix, 2)
    vec = zeros(n_loads)
    for t in axes(pshed_matrix, 1), j in 1:n_loads
        v = pshed_matrix[t, j]
        isnan(v) || (vec[j] += v)
    end
    return (vec = vec,
            α   = NaN,
            n_periods = saved["N_PERIODS"],
            source = path)
end

# ============================================================
# MAX POSSIBLE SHED
# ============================================================
"""
Total kW that could be shed if every load were dropped in every period —
the upper bound on `total_shed` for this case + demand profile. Tries to
read `per_load_period_pd` from a trade-off JLD2 first (always present
when the sweep was run); falls back to `pd_ref_matrix` in a new-schema
bilevel JLD2. Returns NaN if neither source is available.
"""
function _max_possible_shed()
    for fair_func in ("efficiency", "min_max", "palma")
        p = _trade_off_jld2(fair_func)
        p === nothing && continue
        s = JLD2.load(p)
        haskey(s, "per_load_period_pd") || continue
        return sum(s["per_load_period_pd"])
    end
    for fair_func in ("efficiency", "min_max", "palma")
        p = _bilevel_jld2(fair_func)
        p === nothing && continue
        s = JLD2.load(p)
        haskey(s, "pd_ref_matrix") || continue
        return sum(s["pd_ref_matrix"])
    end
    return NaN
end

# ============================================================
# NORMS
# ============================================================
function shed_norms(v::AbstractVector{<:Real})
    finite = filter(isfinite, v)
    isempty(finite) && return (L1 = NaN, L2 = NaN, Linf = NaN, CoV = NaN)
    m = mean(finite)
    s = length(finite) > 1 ? std(finite) : 0.0
    return (
        L1   = norm(finite, 1),
        L2   = norm(finite, 2),
        Linf = norm(finite, Inf),
        CoV  = m > 1e-9 ? s / m : NaN,
    )
end

# ============================================================
# BUILD TABLE
# ============================================================
PROBLEMS   = ["single_level", "bilevel"]
FAIR_FUNCS = ["efficiency", "min_max", "palma"]

max_possible = _max_possible_shed()
@info "Max possible total shed for case=$CASE_KEY: $(round(max_possible, digits=3)) kW"

rows = DataFrame(problem = String[], fair_func = String[],
                 α = Union{Float64,Missing}[], N_periods = Union{Int,Missing}[],
                 L1 = Float64[], L2 = Float64[], Linf = Float64[], CoV = Float64[],
                 pct_max = Float64[],
                 source = String[])

for problem in PROBLEMS, fair_func in FAIR_FUNCS
    path = problem == "single_level" ? _trade_off_jld2(fair_func) : _bilevel_jld2(fair_func)
    if path === nothing
        SKIP_MISSING && continue
        @warn "[$problem / $fair_func] no JLD2 found — skipping row"
        push!(rows, (problem, fair_func, missing, missing,
                     NaN, NaN, NaN, NaN, NaN, "(missing)"))
        continue
    end
    info = problem == "single_level" ?
        _trade_off_shed_vec(path, fair_func) :
        _bilevel_shed_vec(path)
    nm = shed_norms(info.vec)
    pct = isnan(max_possible) || max_possible <= 0 ? NaN : 100.0 * nm.L1 / max_possible
    push!(rows, (problem, fair_func,
                 problem == "single_level" ? info.α : missing,
                 info.n_periods,
                 nm.L1, nm.L2, nm.Linf, nm.CoV, pct,
                 relpath(info.source, results_root)))
end

# ============================================================
# OUTPUT
# ============================================================
println("\nPost-hoc fairness norms — case=$CASE_KEY, pshed_type=$PSHED_TYPE, " *
        "trade-off α target=$(TRADE_OFF_ALPHA_TARGET)")
println("Norms on per-load aggregate shed (sum over periods × phases).")
println("Max possible shed (Σ all loads × all periods) = $(round(max_possible, digits=3)) kW")
println()

# Pretty markdown table. Compute column widths from formatted strings.
fmt_val(v::Real) = isnan(v) ? "—" : @sprintf("%.3f", v)
fmt_pct(v::Real) = isnan(v) ? "—" : @sprintf("%.1f%%", v)

header = ["problem", "fair_func", "L1", "L2", "L∞", "CoV", "%max"]
body_rows = [[
    string(r.problem),
    string(r.fair_func),
    fmt_val(r.L1),
    fmt_val(r.L2),
    fmt_val(r.Linf),
    fmt_val(r.CoV),
    fmt_pct(r.pct_max),
] for r in eachrow(rows)]

widths = [maximum(length, [header[c]; [row[c] for row in body_rows]]) for c in 1:length(header)]
function _line(cells)
    parts = [rpad(string(cells[c]), widths[c]) for c in 1:length(cells)]
    return "| " * join(parts, " | ") * " |"
end
println(_line(header))
println("|" * join(["-"^(widths[c] + 2) for c in 1:length(widths)], "|") * "|")
for row in body_rows
    println(_line(row))
end

println("\nSources (relative to $results_root):")
for r in eachrow(rows)
    println("  $(rpad(r.problem, 12)) $(rpad(r.fair_func, 10)) → $(r.source)")
end

# CSV (sources kept as a separate column).
today = Dates.format(now(), "yyyy-mm-dd")
out_dir = joinpath(results_root, today, "post_hoc_fairness")
mkpath(out_dir)
csv_path = joinpath(out_dir, "$(CASE_KEY)_$(PSHED_TYPE)_fairness_norms.csv")
CSV.write(csv_path, rows)
println("\nCSV → $csv_path")
