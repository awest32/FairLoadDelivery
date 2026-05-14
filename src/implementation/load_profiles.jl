#=
Load profiles for multi-period MLD experiments.

Implements the load-schedule diversification strategy from:

    Hamilton & Aliprantis,
    "Resilient Distribution System Restoration with Equitable Load Shedding,"
    IEEE PECI 2023.

The paper assigns each load one of three 24-hour schedules (Fig. 2 / Table I),
with a per-load ±1-hour time shift for additional diversity. Per-phase nameplate
pd already differs across phases for separately-defined per-phase loads (e.g.
634a/634b/634c, L1/L2/L3), so per-phase profile variation at a bus emerges
naturally from name-deterministic per-load assignment, without artificially
unbalancing 3-phase loads whose pd is balanced in the source data.

For multi-phase loads whose pd vector IS unbalanced (e.g. a hand-edited
.dss with [120, 90, 60] on a 3-phase load), each phase is rotated to a
different (schedule, shift) so the per-phase profile diverges across the day.
=#

using PowerModelsDistribution
const _PMD_LP = PowerModelsDistribution

const LOAD_SCHEDULES = [
    [0.72, 0.70, 0.68, 0.68, 0.70, 0.74, 0.78, 0.80, 0.80, 0.82, 0.82, 0.82,
     0.82, 0.82, 0.80, 0.80, 0.82, 0.84, 0.86, 0.84, 0.82, 0.80, 0.76, 0.74],

    [0.55, 0.52, 0.50, 0.50, 0.52, 0.58, 0.68, 0.80, 0.90, 0.98, 1.02, 1.05,
     1.05, 1.03, 1.00, 0.95, 0.90, 0.82, 0.74, 0.68, 0.62, 0.58, 0.56, 0.55],

    [0.60, 0.58, 0.56, 0.56, 0.58, 0.62, 0.72, 0.78, 0.74, 0.68, 0.65, 0.66,
     0.70, 0.72, 0.74, 0.78, 0.86, 0.96, 1.08, 1.10, 1.02, 0.88, 0.74, 0.66],
]
const N_SCHEDULES     = length(LOAD_SCHEDULES)
const SCHEDULE_LENGTH = length(LOAD_SCHEDULES[1])
const SCHEDULE_SHIFTS = (-1, 0, +1)

"""
    assign_load_profile(load_name) -> (schedule_idx::Int, shift::Int)

Deterministic name→profile mapping. Same name gets the same (schedule, shift)
across runs and across .dss variants that reuse the load name.
"""
function assign_load_profile(load_name::AbstractString)
    h        = hash(load_name)
    sched_id = Int(mod(h, N_SCHEDULES)) + 1
    shift    = SCHEDULE_SHIFTS[Int(mod(h ÷ UInt64(N_SCHEDULES), length(SCHEDULE_SHIFTS))) + 1]
    return sched_id, shift
end

"Schedule value at hour `t` (1-indexed), with a circular `shift` in hours."
function schedule_value(sched_idx::Int, shift::Int, t::Int)
    return LOAD_SCHEDULES[sched_idx][mod1(t - shift, SCHEDULE_LENGTH)]
end

"""
    per_phase_scale_matrix(load_dict, n_periods; peak_stress=1.0) -> Matrix{Float64} (n_phases × n_periods)

Per-phase, per-period scale factor for one math-model load dict.

- single-phase / balanced multi-phase: all phases share the load's primary
  (schedule, shift) profile;
- unbalanced multi-phase (pd entries differ across phases): each phase is
  rotated to (schedule_idx + p - 1, shift + p - 1) so the per-phase profiles
  diverge across the day.

`peak_stress` scales every schedule value uniformly. Paper-faithful schedules
peak at ~1.10, which barely stresses our networks; multi-period scripts pass
`peak_stress > 1` to push peak-hour demand past nameplate and force shedding.
"""
function per_phase_scale_matrix(load_dict::Dict{String,Any}, n_periods::Int;
                                balance_tol::Float64 = 1e-6,
                                peak_stress::Float64 = 1.0)
    @assert n_periods == SCHEDULE_LENGTH "n_periods must equal $SCHEDULE_LENGTH (paper schedules are hourly over 24h)"

    name     = load_dict["name"]
    pd       = load_dict["pd"]
    n_phases = length(pd)

    sched_id, shift = assign_load_profile(name)
    M = zeros(n_phases, n_periods)

    is_balanced = n_phases == 1 ||
        all(abs(pd[p] - pd[1]) ≤ balance_tol * max(abs(pd[1]), 1.0) for p in 1:n_phases)

    if is_balanced
        for t in 1:n_periods
            v = peak_stress * schedule_value(sched_id, shift, t)
            for p in 1:n_phases
                M[p, t] = v
            end
        end
    else
        shift_idx0 = findfirst(==(shift), SCHEDULE_SHIFTS) - 1
        for p in 1:n_phases
            p_sched = mod(sched_id - 1 + (p - 1), N_SCHEDULES) + 1
            p_shift = SCHEDULE_SHIFTS[mod(shift_idx0 + (p - 1), length(SCHEDULE_SHIFTS)) + 1]
            for t in 1:n_periods
                M[p, t] = peak_stress * schedule_value(p_sched, p_shift, t)
            end
        end
    end

    return M
end

"""
    create_multinetwork_data_profiled(base_math, n_periods) -> mn_data

Build a PMD multinetwork data dict where each load's `pd`/`qd` is scaled
per-phase, per-period using `per_phase_scale_matrix(load, n_periods)`.

Drop-in replacement for the uniform-scalar `create_multinetwork_data` in the
single-level `_mn.jl` scripts: each period still has a full deep-copied math
dict, but loads no longer share a single scalar scale.
"""
function create_multinetwork_data_profiled(base_math::Dict{String,Any}, n_periods::Int;
                                            peak_stress::Float64 = 1.0)
    mn_data = Dict{String,Any}(
        "multinetwork" => true,
        "per_unit"     => true,
        "data_model"   => _PMD_LP.MATHEMATICAL,
        "nw"           => Dict{String,Any}(),
    )
    for key in ("baseMVA", "basekv", "bus_lookup", "settings")
        haskey(base_math, key) && (mn_data[key] = deepcopy(base_math[key]))
    end

    load_scales = Dict{String,Matrix{Float64}}()
    base_pd     = Dict{String,Vector{Float64}}()
    base_qd     = Dict{String,Vector{Float64}}()
    for (lid, load) in base_math["load"]
        load_scales[lid] = per_phase_scale_matrix(load, n_periods; peak_stress=peak_stress)
        base_pd[lid]     = copy(load["pd"])
        base_qd[lid]     = copy(load["qd"])
    end

    for t in 1:n_periods
        nw_id = string(t - 1)
        nw_data = deepcopy(base_math)
        delete!(nw_data, "multinetwork")
        for (lid, load) in nw_data["load"]
            scales      = load_scales[lid][:, t]
            load["pd"] = base_pd[lid] .* scales
            load["qd"] = base_qd[lid] .* scales
        end
        nw_data["time_period"] = t
        mn_data["nw"][nw_id]   = nw_data
    end
    return mn_data
end

"""
    aggregate_demand_fraction(base_math, n_periods) -> Vector{Float64}

System-level aggregate scale at each period, weighted by nameplate per-phase
pd. Equals `sum_load_phase(pd_p * scale_p,t) / sum_load_phase(pd_p)` — a
scalar replacement for the old uniform `LOAD_SCALE_FACTORS[t]`, useful for
labeling per-period plots when individual loads now follow distinct schedules.
"""
function aggregate_demand_fraction(base_math::Dict{String,Any}, n_periods::Int)
    total_pd = 0.0
    weighted_t = zeros(n_periods)
    for (_, load) in base_math["load"]
        scales = per_phase_scale_matrix(load, n_periods)
        pd     = load["pd"]
        for p in 1:length(pd)
            total_pd += pd[p]
            for t in 1:n_periods
                weighted_t[t] += pd[p] * scales[p, t]
            end
        end
    end
    return total_pd > 0 ? weighted_t ./ total_pd : zeros(n_periods)
end

"""
    profile_assignment_table(base_math) -> Vector of NamedTuples

Diagnostic: returns the (load_name, n_phases, balanced, sched_idx, shift) for
each load. Handy for sanity-checking which loads got which schedule before
running an expensive sweep.
"""
function profile_assignment_table(base_math::Dict{String,Any})
    rows = NamedTuple[]
    for (_, load) in base_math["load"]
        name = load["name"]
        pd   = load["pd"]
        n_ph = length(pd)
        is_balanced = n_ph == 1 ||
            all(abs(pd[p] - pd[1]) ≤ 1e-6 * max(abs(pd[1]), 1.0) for p in 1:n_ph)
        sched_id, shift = assign_load_profile(name)
        push!(rows, (name=name, n_phases=n_ph, balanced=is_balanced,
                     sched=sched_id, shift=shift))
    end
    return sort(rows; by = r -> r.name)
end
