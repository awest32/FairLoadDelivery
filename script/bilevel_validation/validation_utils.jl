"""
    Validation utilities for the bilevel FLDP experiment.

    Provides functions to check:
    - Voltage limits on energized buses
    - Switch current (ampacity) limits
    - Data label consistency across pipeline stages
    - AC power flow feasibility with correct switch ratings
"""

# ============================================================
# CONSTANTS
# ============================================================
const V_MIN = 0.9
const V_MAX = 1.1
const V_MIN_SQ = V_MIN^2  # 0.81
const V_MAX_SQ = V_MAX^2  # 1.21
const TOLERANCE = 1e-6
# ============================================================
# VOLTAGE LIMIT CHECKS
# ============================================================

"""
    check_voltage_limits_relaxed(result::Dict, math::Dict)

Check voltage limits for a relaxed (LinDist3Flow) solution.
Uses w = v² from the solution. Skips de-energized buses (w ≈ 0).
Returns (passed::Bool, violations::Vector, summary::Dict)
"""
function check_voltage_limits_relaxed(result::Dict, math::Dict)
    violations = []
    n_checked = 0
    n_skipped = 0

    if !haskey(result, "solution") || !haskey(result["solution"], "bus")
        return false, ["No bus solution found"], Dict("checked" => 0, "skipped" => 0, "violations" => 0)
    end

    for (bus_id, bus_soln) in result["solution"]["bus"]
        if !haskey(bus_soln, "w")
            continue
        end
        w_vals = bus_soln["w"]

        if w_vals isa Dict
            for (phase, w) in w_vals
                w_scalar = w isa AbstractArray ? first(w) : Float64(w)
                if w_scalar < 1e-6
                    n_skipped += 1
                    continue
                end
                n_checked += 1
                if w_scalar < V_MIN_SQ - TOLERANCE
                    push!(violations, (bus_id=bus_id, phase=phase, w=w_scalar, type="undervoltage", limit=V_MIN_SQ))
                elseif w_scalar > V_MAX_SQ + TOLERANCE
                    push!(violations, (bus_id=bus_id, phase=phase, w=w_scalar, type="overvoltage", limit=V_MAX_SQ))
                end
            end
        elseif w_vals isa AbstractArray
            for (idx, w) in enumerate(w_vals)
                w_scalar = Float64(w)
                if w_scalar < 1e-6
                    n_skipped += 1
                    continue
                end
                n_checked += 1
                if w_scalar < V_MIN_SQ - TOLERANCE
                    push!(violations, (bus_id=bus_id, phase=idx, w=w_scalar, type="undervoltage", limit=V_MIN_SQ))
                elseif w_scalar > V_MAX_SQ + TOLERANCE
                    push!(violations, (bus_id=bus_id, phase=idx, w=w_scalar, type="overvoltage", limit=V_MAX_SQ))
                end
            end
        else
            w_scalar = Float64(w_vals)
            if w_scalar < 1e-6
                n_skipped += 1
                continue
            end
            n_checked += 1
            if w_scalar < V_MIN_SQ
                push!(violations, (bus_id=bus_id, phase="all", w=w_scalar, type="undervoltage", limit=V_MIN_SQ-TOLERANCE))
            elseif w_scalar > V_MAX_SQ
                push!(violations, (bus_id=bus_id, phase="all", w=w_scalar, type="overvoltage", limit=V_MAX_SQ+TOLERANCE))
            end
        end
    end

    summary = Dict("checked" => n_checked, "skipped_deenergized" => n_skipped, "violations" => length(violations))
    return isempty(violations), violations, summary
end

"""
    check_voltage_limits_ac(result::Dict, math::Dict)

Check voltage limits for an AC (IVR) power flow solution.
Uses vr² + vi² to compute voltage magnitude squared.
Skips de-energized buses.
Returns (passed::Bool, violations::Vector, summary::Dict)
"""
function check_voltage_limits_ac(result::Dict, math::Dict)
    violations = []
    n_checked = 0
    n_skipped = 0

    if !haskey(result, "solution") || !haskey(result["solution"], "bus")
        return false, ["No bus solution found"], Dict("checked" => 0, "skipped" => 0, "violations" => 0)
    end

    for (bus_id, bus_soln) in result["solution"]["bus"]
        vr = get(bus_soln, "vr", nothing)
        vi = get(bus_soln, "vi", nothing)
        if vr === nothing || vi === nothing
            continue
        end

        # Compute v² = vr² + vi² per phase
        if vr isa AbstractArray && vi isa AbstractArray
            for idx in 1:length(vr)
                v_sq = vr[idx]^2 + vi[idx]^2
                if v_sq < 1e-6
                    n_skipped += 1
                    continue
                end
                n_checked += 1
                if v_sq < V_MIN_SQ - TOLERANCE
                    push!(violations, (bus_id=bus_id, phase=idx, v_sq=v_sq, v_mag=sqrt(v_sq), type="undervoltage", limit=V_MIN_SQ))
                elseif v_sq > V_MAX_SQ + TOLERANCE
                    push!(violations, (bus_id=bus_id, phase=idx, v_sq=v_sq, v_mag=sqrt(v_sq), type="overvoltage", limit=V_MAX_SQ))
                end
            end
        end
    end

    summary = Dict("checked" => n_checked, "skipped_deenergized" => n_skipped, "violations" => length(violations))
    return isempty(violations), violations, summary
end

# ============================================================
# CURRENT (AMPACITY) LIMIT CHECKS
# ============================================================

"""
    check_switch_ampacity(result::Dict, math::Dict)

Check switch current limits using the constraint: P² + Q² ≤ z_switch * w * I_rating²
Only checks closed switches with finite current ratings.
Returns (passed::Bool, violations::Vector, summary::Dict)
"""
function check_switch_ampacity(result::Dict, math::Dict)
    violations = []
    n_checked = 0
    n_skipped_open = 0
    n_skipped_no_rating = 0
    utilizations = Dict{String, Float64}()

    if !haskey(result, "solution") || !haskey(result["solution"], "switch")
        return true, [], Dict("checked" => 0, "violations" => 0, "message" => "No switch solution")
    end

    for (switch_id, switch) in math["switch"]
        if !haskey(result["solution"]["switch"], switch_id)
            continue
        end

        switch_soln = result["solution"]["switch"][switch_id]

        # Get switch state
        state = get(switch_soln, "state", 1.0)
        z_switch = state isa AbstractArray ? minimum(state) : Float64(state)

        if z_switch < 1e-6
            n_skipped_open += 1
            continue
        end

        # Get current rating
        current_rating = get(switch, "current_rating", [Inf])
        if all(r -> r == Inf || r <= 0, current_rating)
            n_skipped_no_rating += 1
            continue
        end

        # Get power flow
        pf = get(switch_soln, "psw_fr", get(switch_soln, "pf", [0.0]))
        qf = get(switch_soln, "qsw_fr", get(switch_soln, "qf", [0.0]))

        # Get voltage at from bus
        f_bus = switch["f_bus"]
        f_connections = get(switch, "f_connections", [1, 2, 3])

        w_vals = nothing
        if haskey(result["solution"], "bus") && haskey(result["solution"]["bus"], string(f_bus))
            bus_soln = result["solution"]["bus"][string(f_bus)]
            if haskey(bus_soln, "w")
                w_vals = bus_soln["w"]
            end
        end

        # Convert to arrays
        pf_arr = pf isa Dict ? collect(values(pf)) : (pf isa AbstractArray ? pf : [pf])
        qf_arr = qf isa Dict ? collect(values(qf)) : (qf isa AbstractArray ? qf : [qf])

        max_util = 0.0
        for (idx, (p, q)) in enumerate(zip(pf_arr, qf_arr))
            s_squared = p^2 + q^2

            # Get w for this phase
            conn = idx <= length(f_connections) ? f_connections[idx] : idx
            w = 1.0
            if w_vals !== nothing
                if w_vals isa Dict
                    if haskey(w_vals, conn)
                        w = w_vals[conn]
                    elseif haskey(w_vals, string(conn))
                        w = w_vals[string(conn)]
                    else
                        w_list = collect(values(w_vals))
                        if idx <= length(w_list)
                            w = w_list[idx]
                        end
                    end
                elseif w_vals isa AbstractArray && idx <= length(w_vals)
                    w = w_vals[idx]
                else
                    w = w_vals isa Number ? Float64(w_vals) : 1.0
                end
            end
            w = w isa AbstractArray ? first(w) : Float64(w)

            rating = idx <= length(current_rating) ? current_rating[idx] : current_rating[1]

            if rating > 0 && rating < Inf && w > 1e-6
                n_checked += 1
                # Constraint: S² ≤ z_switch * w * I²
                limit = z_switch * w * rating^2
                util = sqrt(s_squared) / (sqrt(z_switch * w) * rating) * 100
                max_util = max(max_util, util)

                if s_squared > limit
                    push!(violations, (switch_id=switch_id, phase=idx,
                        s_squared=s_squared, limit=limit,
                        utilization_pct=util, z_switch=z_switch, w=w, rating=rating))
                end
            end
        end
        utilizations[switch_id] = max_util
    end

    summary = Dict(
        "checked" => n_checked,
        "skipped_open" => n_skipped_open,
        "skipped_no_rating" => n_skipped_no_rating,
        "violations" => length(violations),
        "utilizations" => utilizations
    )
    return isempty(violations), violations, summary
end

# ============================================================
# DATA LABEL CONSISTENCY CHECKS
# ============================================================

"""
    check_label_consistency(math::Dict, pshed_ids, weight_ids, stage::String)

Verify that pshed_ids and weight_ids correspond to valid load IDs in the math dictionary.
Returns (passed::Bool, issues::Vector)
"""
function check_label_consistency(math::Dict, pshed_ids, weight_ids, stage::String)
    issues = []
    math_load_ids = Set(parse(Int, k) for k in keys(math["load"]))

    # Check pshed_ids
    for id in pshed_ids
        if !(id in math_load_ids)
            push!(issues, "[$stage] pshed_id=$id not found in math[\"load\"]")
        end
    end

    # Check weight_ids
    for id in weight_ids
        if !(id in math_load_ids)
            push!(issues, "[$stage] weight_id=$id not found in math[\"load\"]")
        end
    end

    # Check that pshed_ids and weight_ids refer to the same loads
    if Set(pshed_ids) != Set(weight_ids)
        push!(issues, "[$stage] pshed_ids and weight_ids refer to different load sets: pshed=$(sort(collect(pshed_ids))) vs weight=$(sort(collect(weight_ids)))")
    end

    # Check ordering consistency
    if collect(pshed_ids) != collect(weight_ids)
        push!(issues, "[$stage] pshed_ids and weight_ids have different ordering (sets match but order differs)")
    end

    return isempty(issues), issues
end

"""
    check_switch_label_consistency(math::Dict, switch_states::Dict, stage::String)

Verify that switch state keys correspond to valid switch IDs in the math dictionary.
Returns (passed::Bool, issues::Vector)
"""
function check_switch_label_consistency(math::Dict, switch_states::Dict, stage::String)
    issues = []
    math_switch_ids = Set(parse(Int, k) for k in keys(math["switch"]))

    for s_id in keys(switch_states)
        if !(s_id in math_switch_ids)
            push!(issues, "[$stage] switch_id=$s_id not found in math[\"switch\"]")
        end
    end

    # Check if any math switches are missing from the states
    for s_id in math_switch_ids
        if !haskey(switch_states, s_id)
            push!(issues, "[$stage] math switch $s_id not present in extracted switch_states")
        end
    end

    return isempty(issues), issues
end

"""
    check_block_label_consistency(math::Dict, block_status::Dict, stage::String)

Verify that block status keys correspond to valid block IDs in the math dictionary.
Returns (passed::Bool, issues::Vector)
"""
function check_block_label_consistency(math::Dict, block_status::Dict, stage::String)
    issues = []
    math_block_ids = Set(parse(Int, k) for k in keys(math["block"]))

    for b_id in keys(block_status)
        if !(b_id in math_block_ids)
            push!(issues, "[$stage] block_id=$b_id not found in math[\"block\"]")
        end
    end

    return isempty(issues), issues
end

"""
    check_weight_update_consistency(math_before::Dict, math_after::Dict, weight_ids, weight_vals)

Verify that weight updates were applied correctly to the math dictionary.
Returns (passed::Bool, issues::Vector)
"""
function check_weight_update_consistency(math_before::Dict, math_after::Dict, weight_ids, weight_vals)
    issues = []

    for (i, w_id) in enumerate(weight_ids)
        key = string(w_id)
        if !haskey(math_after["load"], key)
            push!(issues, "Load $w_id not found in updated math dict")
            continue
        end

        actual_weight = math_after["load"][key]["weight"]
        expected_weight = weight_vals[i]

        if abs(actual_weight - expected_weight) > 1e-10
            push!(issues, "Load $w_id: expected weight=$expected_weight, got $actual_weight")
        end
    end

    return isempty(issues), issues
end

# ============================================================
# REPORTING
# ============================================================

"""
    print_validation_header(stage::String)

Print a formatted header for a validation stage.
"""
function print_validation_header(stage::String)
    println("\n" * "="^70)
    println("  VALIDATION: $stage")
    println("="^70)
end

"""
    print_check_result(check_name::String, passed::Bool, details::String="")

Print the result of a validation check.
"""
function print_check_result(check_name::String, passed::Bool, details::String="")
    status = passed ? "PASS" : "FAIL"
    marker = passed ? "[+]" : "[X]"
    println("  $marker $check_name: $status")
    if !passed && details != ""
        for line in split(details, "\n")
            println("      $line")
        end
    end
end

"""
    generate_summary_report(results::Dict, save_path::String)

Generate a summary report of all validation checks.
Writes to both stdout and a file.
"""
function generate_summary_report(results::Dict, save_path::String)
    report_lines = String[]

    push!(report_lines, "\n" * "="^70)
    push!(report_lines, "  BILEVEL VALIDATION SUMMARY REPORT")
    push!(report_lines, "="^70)
    push!(report_lines, "  Case: $(get(results, "case", "unknown"))")
    push!(report_lines, "  Fairness Function: $(get(results, "fair_func", "unknown"))")
    push!(report_lines, "  Iterations: $(get(results, "iterations", "unknown"))")
    push!(report_lines, "")

    total_checks = 0
    total_passed = 0
    total_failed = 0

    for stage in ["setup", "relaxed_fldp", "random_rounding", "ac_feasibility"]
        if !haskey(results, stage)
            continue
        end
        stage_results = results[stage]
        push!(report_lines, "  --- $stage ---")

        for (check_name, check_result) in stage_results
            total_checks += 1
            passed = check_result["passed"]
            if passed
                total_passed += 1
                push!(report_lines, "    [+] $check_name: PASS")
            else
                total_failed += 1
                push!(report_lines, "    [X] $check_name: FAIL")
                if haskey(check_result, "details")
                    for d in check_result["details"]
                        push!(report_lines, "        - $d")
                    end
                end
            end
        end
        push!(report_lines, "")
    end

    # AC Feasibility Summary
    if haskey(results, "ac_feasibility_summary")
        ac_sum = results["ac_feasibility_summary"]
        push!(report_lines, "  --- AC FEASIBILITY SUMMARY ---")
        push!(report_lines, "    Convergence: $(ac_sum["converged"] ? "YES" : "NO")")
        push!(report_lines, "    Termination Status: $(ac_sum["termination_status"])")
        if ac_sum["converged"]
            push!(report_lines, "    Voltage Violations: $(ac_sum["voltage_violations"])")
            push!(report_lines, "    Current Rating Violations: $(ac_sum["current_violations"])")
            if ac_sum["voltage_violations"] > 0 || ac_sum["current_violations"] > 0
                push!(report_lines, "    ** AC SOLUTION IS NOT FEASIBLE WITH CORRECT RATINGS **")
            else
                push!(report_lines, "    AC solution is feasible with correct voltage and current ratings.")
            end
        else
            push!(report_lines, "    ** NO AC POWER FLOW FEASIBLE SOLUTION FOUND **")
        end
        push!(report_lines, "")
    end

    push!(report_lines, "  TOTALS: $total_checks checks, $total_passed passed, $total_failed failed")
    push!(report_lines, "="^70)

    # Print to stdout
    for line in report_lines
        println(line)
    end

    # Write to file
    mkpath(dirname(save_path))
    open(save_path, "w") do io
        for line in report_lines
            println(io, line)
        end
    end
    println("\n  Report saved to: $save_path")
end
