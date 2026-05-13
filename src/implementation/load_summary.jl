"""
    load_summary_rows(math, pshed_per_load_kw; extra=NamedTuple()) -> DataFrame

Per-bus + network-total load served / shed for a single run. `pshed_per_load_kw`
maps a load id (Int or String) to the total kW shed at that load (summed over
phases). Returns one row per bus plus a `bus = "__network_total__"` row, with
columns: pd_kw, served_kw, shed_kw, served_pct, shed_pct.

`extra` is a NamedTuple of scenario columns (e.g. `(alpha=0.5, period=3)`)
prepended to every row, so many runs can be concatenated into one tidy CSV.
"""
function load_summary_rows(math::Dict{String,Any}, pshed_per_load_kw; extra::NamedTuple=NamedTuple())
    bus_name_of = Dict(parse(Int, b) => split(string(bus["source_id"]), ".")[end]
                       for (b, bus) in math["bus"])

    bus_pd   = Dict{String, Float64}()
    bus_shed = Dict{String, Float64}()
    for (lid_str, load) in math["load"]
        bn  = bus_name_of[load["load_bus"]]
        lid = parse(Int, lid_str)
        bus_pd[bn]   = get(bus_pd,   bn, 0.0) + sum(load["pd"])
        bus_shed[bn] = get(bus_shed, bn, 0.0) +
                       Float64(get(pshed_per_load_kw, lid_str, get(pshed_per_load_kw, lid, 0.0)))
    end

    df = DataFrames.DataFrame(bus = String[], pd_kw = Float64[], served_kw = Float64[],
                              shed_kw = Float64[], served_pct = Float64[], shed_pct = Float64[])
    for bn in sort(collect(keys(bus_pd)))
        pd, sh = bus_pd[bn], bus_shed[bn]
        push!(df, (bn, pd, pd - sh, sh,
                   pd > 0 ? 100 * (pd - sh) / pd : NaN,
                   pd > 0 ? 100 * sh / pd : NaN))
    end
    tpd, tsh = sum(values(bus_pd)), sum(values(bus_shed))
    push!(df, ("__network_total__", tpd, tpd - tsh, tsh,
               tpd > 0 ? 100 * (tpd - tsh) / tpd : NaN,
               tpd > 0 ? 100 * tsh / tpd : NaN))

    if !isempty(extra)
        n = DataFrames.nrow(df)
        df = hcat(DataFrames.DataFrame((k => fill(v, n) for (k, v) in pairs(extra))...), df)
    end
    return df
end

"""
    append_load_summary!(path, df)

Write `df` to `path`, appending without re-writing the header if the file
already exists. Use to accumulate per-alpha / per-period rows into a single
CSV per script.
"""
function append_load_summary!(path::String, df::DataFrames.DataFrame)
    exists = isfile(path)
    CSV.write(path, df; append=exists, writeheader=!exists)
    return path
end

"""
    load_shed_metrics(math, pshed_per_load_kw; extra=NamedTuple()) -> (DataFrame, Dict)

Distribution-level summary of a single run's load shedding. The distribution
is the per-load shed vector (one entry per `math["load"]`, summed over phases).
Returns a one-row `DataFrame` (for CSV accumulation) and a `Dict` of the same
metrics keyed by `Symbol`:

  - `total_pd_kw`, `total_shed_kw`, `total_served_kw`
  - `pct_shed`, `pct_served`
  - `l1_norm`, `l2_norm`, `linf_norm` on the per-load shed vector
  - `cv` — coefficient of variation (std / mean) of the per-load shed vector;
    `NaN` when the mean is zero.

`extra` is a NamedTuple of scenario columns (e.g. `(alpha=0.5, period=3)`)
prepended to the row, so many runs concatenate into one tidy CSV.
"""
function load_shed_metrics(math::Dict{String,Any}, pshed_per_load_kw; extra::NamedTuple=NamedTuple())
    shed = Float64[]
    pd_total = 0.0
    for (lid_str, load) in math["load"]
        lid = parse(Int, lid_str)
        push!(shed, Float64(get(pshed_per_load_kw, lid_str, get(pshed_per_load_kw, lid, 0.0))))
        pd_total += sum(load["pd"])
    end

    n            = length(shed)
    total_shed   = sum(shed)
    total_served = pd_total - total_shed
    pct_shed     = pd_total > 0 ? 100 * total_shed   / pd_total : NaN
    pct_served   = pd_total > 0 ? 100 * total_served / pd_total : NaN
    l1           = LinearAlgebra.norm(shed, 1)
    l2           = LinearAlgebra.norm(shed, 2)
    linf         = LinearAlgebra.norm(shed, Inf)
    μ            = n > 0 ? total_shed / n : NaN
    σ            = n > 1 ? Statistics.std(shed; corrected=true) : 0.0
    cv           = (isfinite(μ) && μ > 0) ? σ / μ : NaN

    metrics = Dict{Symbol,Float64}(
        :total_pd_kw     => pd_total,
        :total_shed_kw   => total_shed,
        :total_served_kw => total_served,
        :pct_shed        => pct_shed,
        :pct_served      => pct_served,
        :l1_norm         => l1,
        :l2_norm         => l2,
        :linf_norm       => linf,
        :cv              => cv,
    )

    cols = [:total_pd_kw, :total_shed_kw, :total_served_kw, :pct_shed, :pct_served,
            :l1_norm, :l2_norm, :linf_norm, :cv]
    df = DataFrames.DataFrame((c => [metrics[c]] for c in cols)...)
    if !isempty(extra)
        df = hcat(DataFrames.DataFrame((k => [v] for (k, v) in pairs(extra))...), df)
    end
    return df, metrics
end
