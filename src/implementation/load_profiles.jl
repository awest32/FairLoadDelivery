#=
Load profiles for multi-period MLD experiments.

Implements the load-schedule diversification strategy from:

    Hamilton & Aliprantis,
    "Resilient Distribution System Restoration with Equitable Load Shedding,"
    IEEE PECI 2023.

The paper assigns each load one of three 24-hour schedules (Fig. 2 / Table I),
with a per-load ±1-hour time shift for additional diversity. We layer a
bus-aware step on top: `assign_profiles_by_bus` guarantees that no two phases
served at the same bus share the same `(schedule, shift)` pair. Allowed at any
bus: same schedule + different shifts, different schedules + same shift, or
different schedules + different shifts.

- Balanced multi-phase loads: keep the load's hashed schedule, rotate the shift
  across phases so per-phase profiles diverge in time.
- Unbalanced multi-phase loads (pd entries differ across phases, e.g. a
  hand-edited [120, 90, 60] 3-phase load): rotate both schedule and shift.
- Collisions with other loads at the same bus: resolved by bumping the shift
  first (preserving the schedule when possible) and the schedule only if every
  shift on that schedule is already taken.
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
# Daily mean of each raw schedule — used when `center_at_nominal=true` so that
# the *daily-average* per-load scale equals 1.0× nominal pd (peaks above, troughs
# below). Without it, the paper schedules average ~0.75× nominal and only barely
# touch 1.0× at peak, making nameplate pd effectively the daily peak rather
# than the daily mean.
const SCHEDULE_MEANS = [sum(s) / length(s) for s in LOAD_SCHEDULES]

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
Walk (schedule, shift) starting at `(init_sched, init_shift)` and return the
first pair not already in `used`. Bumps shift first (preserving schedule when
possible), then bumps schedule. Falls back to the initial pair if every one of
the `N_SCHEDULES * length(SCHEDULE_SHIFTS)` combinations is taken.
"""
function _next_free_profile(init_sched::Int, init_shift::Int,
                            used::AbstractSet{Tuple{Int,Int}})
    init_shift_idx = findfirst(==(init_shift), SCHEDULE_SHIFTS) - 1
    for sched_off in 0:(N_SCHEDULES - 1)
        s = mod(init_sched - 1 + sched_off, N_SCHEDULES) + 1
        for shift_off in 0:(length(SCHEDULE_SHIFTS) - 1)
            sh = SCHEDULE_SHIFTS[mod(init_shift_idx + shift_off, length(SCHEDULE_SHIFTS)) + 1]
            (s, sh) in used || return (s, sh)
        end
    end
    return (init_sched, init_shift)
end

"""
    assign_profiles_by_bus(base_math; balance_tol=1e-6)
        -> Dict{String, Vector{Tuple{Int,Int}}}

Per-bus, per-phase `(schedule_idx, shift)` assignment such that no two phases
served at the same bus share the same `(schedule, shift)` pair. Allowed at any
bus: same schedule + different shifts, different schedules + same shift, or
different schedules + different shifts.

Initial pick per phase reuses the per-load hash from `assign_load_profile`
(balanced multi-phase loads: keep the schedule, rotate the shift across phases;
unbalanced multi-phase loads: rotate both). Collisions with other loads at the
same bus are then resolved by `_next_free_profile`, preferring to bump shift
before schedule.
"""
function assign_profiles_by_bus(base_math::Dict{String,Any};
                                balance_tol::Float64 = 1e-6)
    bus_to_lids = Dict{Int, Vector{String}}()
    for (lid, load) in base_math["load"]
        push!(get!(bus_to_lids, load["load_bus"], String[]), lid)
    end

    assignment = Dict{String, Vector{Tuple{Int,Int}}}()
    for bus in sort(collect(keys(bus_to_lids)))
        used = Set{Tuple{Int,Int}}()
        for lid in sort(bus_to_lids[bus])
            load     = base_math["load"][lid]
            pd       = load["pd"]
            n_phases = length(pd)

            base_sched, base_shift = assign_load_profile(load["name"])
            shift_idx0 = findfirst(==(base_shift), SCHEDULE_SHIFTS) - 1

            is_balanced = n_phases == 1 ||
                all(abs(pd[p] - pd[1]) ≤ balance_tol * max(abs(pd[1]), 1.0) for p in 1:n_phases)

            phases = Vector{Tuple{Int,Int}}(undef, n_phases)
            for p in 1:n_phases
                init_sched = is_balanced ? base_sched :
                    mod(base_sched - 1 + (p - 1), N_SCHEDULES) + 1
                init_shift = SCHEDULE_SHIFTS[mod(shift_idx0 + (p - 1), length(SCHEDULE_SHIFTS)) + 1]
                phases[p]  = _next_free_profile(init_sched, init_shift, used)
                push!(used, phases[p])
            end
            assignment[lid] = phases
        end
    end
    return assignment
end

"""
    per_phase_scale_matrix(load_dict, n_periods; hours=0:SCHEDULE_LENGTH-1,
                           phase_profile=nothing, peak_stress=1.0,
                           center_at_nominal=false)
        -> Matrix{Float64} (n_phases × n_periods)

Per-phase, per-period scale factor for one math-model load dict.

`hours` is a 0-indexed selection of hours-of-day in `0:SCHEDULE_LENGTH-1` and
defines which periods are sampled (default = the full 24-hour day). Pass a
shorter `hours` (e.g. `[2, 8, 12, 15, 18, 21]`) to downsample. `n_periods` must
equal `length(hours)` — kept as a positional arg so call sites stay explicit
about the per-period array length downstream.

Pass `phase_profile` (from `assign_profiles_by_bus`) for bus-aware assignment
where phases at the same bus are guaranteed to differ in `(schedule, shift)`.
If omitted, falls back to a per-load-only rotation: balanced multi-phase loads
keep the load's schedule and rotate shift across phases; unbalanced multi-phase
loads rotate both. The fallback diversifies within a load but cannot detect
collisions with other loads at the same bus — prefer the bus-aware path.

`peak_stress` scales every schedule value uniformly. When `center_at_nominal`
is false (default), the raw paper schedules are used — their daily mean is
~0.75× nominal and peaks barely reach 1.0× nominal, so multi-period scripts
typically pass `peak_stress > 1` to force shedding. When `center_at_nominal`
is true, each schedule is first divided by its own daily mean so the daily-
average scale equals `peak_stress` exactly (and the nameplate pd is the daily
*mean* rather than the daily peak). This makes the multi-period mean comparable
to the single-period nominal load.
"""
function per_phase_scale_matrix(load_dict::Dict{String,Any}, n_periods::Int;
                                hours::AbstractVector{Int} = 0:SCHEDULE_LENGTH-1,
                                balance_tol::Float64 = 1e-6,
                                peak_stress::Float64 = 1.0,
                                center_at_nominal::Bool = false,
                                phase_profile::Union{Nothing,Vector{Tuple{Int,Int}}} = nothing)
    @assert length(hours) == n_periods "length(hours) ($(length(hours))) must equal n_periods ($n_periods)"
    @assert all(0 <= h < SCHEDULE_LENGTH for h in hours) "hours must be 0-indexed in 0:$(SCHEDULE_LENGTH-1)"

    pd       = load_dict["pd"]
    n_phases = length(pd)

    profile = phase_profile === nothing ?
        _per_load_phase_profile(load_dict; balance_tol=balance_tol) :
        phase_profile
    @assert length(profile) == n_phases "phase_profile length ($(length(profile))) must equal n_phases ($n_phases)"

    M = zeros(n_phases, n_periods)
    for p in 1:n_phases
        p_sched, p_shift = profile[p]
        norm = center_at_nominal ? SCHEDULE_MEANS[p_sched] : 1.0
        for (t, h) in enumerate(hours)
            # schedule_value expects 1-indexed t; `hours` is 0-indexed hour-of-day.
            M[p, t] = peak_stress * schedule_value(p_sched, p_shift, h + 1) / norm
        end
    end
    return M
end

"Fallback per-load phase profile when no bus-aware assignment is provided."
function _per_load_phase_profile(load_dict::Dict{String,Any};
                                 balance_tol::Float64 = 1e-6)
    pd       = load_dict["pd"]
    n_phases = length(pd)
    base_sched, base_shift = assign_load_profile(load_dict["name"])
    shift_idx0 = findfirst(==(base_shift), SCHEDULE_SHIFTS) - 1
    is_balanced = n_phases == 1 ||
        all(abs(pd[p] - pd[1]) ≤ balance_tol * max(abs(pd[1]), 1.0) for p in 1:n_phases)

    profile = Vector{Tuple{Int,Int}}(undef, n_phases)
    for p in 1:n_phases
        p_sched = is_balanced ? base_sched :
            mod(base_sched - 1 + (p - 1), N_SCHEDULES) + 1
        p_shift = SCHEDULE_SHIFTS[mod(shift_idx0 + (p - 1), length(SCHEDULE_SHIFTS)) + 1]
        profile[p] = (p_sched, p_shift)
    end
    return profile
end

"""
    create_multinetwork_data_profiled(base_math, n_periods;
                                       hours=0:SCHEDULE_LENGTH-1, ...) -> mn_data

Build a PMD multinetwork data dict where each load's `pd`/`qd` is scaled
per-phase, per-period using `per_phase_scale_matrix(load, n_periods; hours)`.

`hours` (0-indexed in `0:SCHEDULE_LENGTH-1`) selects which hours-of-day appear
as periods. Default = the full 24-hour day; pass a shorter vector (e.g.
`[2, 8, 12, 15, 18, 21]`) to downsample. `n_periods` must match `length(hours)`.

Drop-in replacement for the uniform-scalar `create_multinetwork_data` in the
single-level `_mn.jl` scripts: each period still has a full deep-copied math
dict, but loads no longer share a single scalar scale.
"""
function create_multinetwork_data_profiled(base_math::Dict{String,Any}, n_periods::Int;
                                            hours::AbstractVector{Int} = 0:SCHEDULE_LENGTH-1,
                                            peak_stress::Float64 = 1.0,
                                            center_at_nominal::Bool = false)
    mn_data = Dict{String,Any}(
        "multinetwork" => true,
        "per_unit"     => true,
        "data_model"   => _PMD_LP.MATHEMATICAL,
        "nw"           => Dict{String,Any}(),
    )
    for key in ("baseMVA", "basekv", "bus_lookup", "settings")
        haskey(base_math, key) && (mn_data[key] = deepcopy(base_math[key]))
    end

    phase_profiles = assign_profiles_by_bus(base_math)
    load_scales = Dict{String,Matrix{Float64}}()
    base_pd     = Dict{String,Vector{Float64}}()
    base_qd     = Dict{String,Vector{Float64}}()
    for (lid, load) in base_math["load"]
        load_scales[lid] = per_phase_scale_matrix(load, n_periods;
            hours=hours, peak_stress=peak_stress, center_at_nominal=center_at_nominal,
            phase_profile=phase_profiles[lid])
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
    aggregate_demand_fraction(base_math, n_periods;
                               hours=0:SCHEDULE_LENGTH-1, ...) -> Vector{Float64}

System-level aggregate scale at each period, weighted by nameplate per-phase
pd. Equals `sum_load_phase(pd_p * scale_p,t) / sum_load_phase(pd_p)` — a
scalar replacement for the old uniform `LOAD_SCALE_FACTORS[t]`, useful for
labeling per-period plots when individual loads now follow distinct schedules.

`hours` (0-indexed in `0:SCHEDULE_LENGTH-1`) selects which hours-of-day are
sampled; default = full 24-hour day. `n_periods` must match `length(hours)`.
"""
function aggregate_demand_fraction(base_math::Dict{String,Any}, n_periods::Int;
                                   hours::AbstractVector{Int} = 0:SCHEDULE_LENGTH-1,
                                   center_at_nominal::Bool = false)
    phase_profiles = assign_profiles_by_bus(base_math)
    total_pd = 0.0
    weighted_t = zeros(n_periods)
    for (lid, load) in base_math["load"]
        scales = per_phase_scale_matrix(load, n_periods;
            hours=hours, center_at_nominal=center_at_nominal,
            phase_profile=phase_profiles[lid])
        pd = load["pd"]
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

Diagnostic: returns the post-collision-resolution
`(load_name, bus, n_phases, balanced, phases::Vector{(sched, shift)})` for
each load. Use it to sanity-check the bus-aware assignment — phases sharing a
bus should never share a `(sched, shift)` pair.
"""
function profile_assignment_table(base_math::Dict{String,Any})
    phase_profiles = assign_profiles_by_bus(base_math)
    rows = NamedTuple[]
    for (lid, load) in base_math["load"]
        name = load["name"]
        pd   = load["pd"]
        n_ph = length(pd)
        is_balanced = n_ph == 1 ||
            all(abs(pd[p] - pd[1]) ≤ 1e-6 * max(abs(pd[1]), 1.0) for p in 1:n_ph)
        push!(rows, (name=name, bus=load["load_bus"], n_phases=n_ph,
                     balanced=is_balanced, phases=phase_profiles[lid]))
    end
    return sort(rows; by = r -> (r.bus, r.name))
end
