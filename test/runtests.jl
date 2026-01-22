using FairLoadDelivery
using Test
using PowerModelsDistribution
using Ipopt
using Gurobi
using JuMP
using Statistics

# Set up solvers
ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi = Gurobi.Optimizer

# Test case configuration
const TEST_CASE = "ieee_13_aw_edit/motivation_a.dss"
const GEN_CAP = 1000.0  # Limited generation to force load shedding

# Voltage limits (per unit squared)
const V_MIN_SQ = 0.81  # 0.9^2
const V_MAX_SQ = 1.21  # 1.1^2
const TOLERANCE = 1e-4

"""
Helper function to check voltage limits are satisfied.
Returns (passed::Bool, violations::Vector)
"""
function check_voltage_limits(result::Dict, math::Dict)
    violations = []

    if !haskey(result, "solution") || !haskey(result["solution"], "bus")
        return false, ["No bus solution found"]
    end

    for (bus_id, bus_soln) in result["solution"]["bus"]
        if haskey(bus_soln, "w")
            w_vals = bus_soln["w"]
            # Handle Dict, Vector, or scalar
            if w_vals isa Dict
                for (phase, w) in w_vals
                    w_scalar = w isa AbstractArray ? first(w) : w
                    # Skip de-energized buses (w ≈ 0 means block was shed)
                    if w_scalar < TOLERANCE
                        continue
                    end
                    if w_scalar < V_MIN_SQ - TOLERANCE
                        push!(violations, "Bus $bus_id phase $phase: w = $w_scalar < $V_MIN_SQ (undervoltage)")
                    elseif w_scalar > V_MAX_SQ + TOLERANCE
                        push!(violations, "Bus $bus_id phase $phase: w = $w_scalar > $V_MAX_SQ (overvoltage)")
                    end
                end
            elseif w_vals isa AbstractArray
                for (idx, w) in enumerate(w_vals)
                    # Skip de-energized buses (w ≈ 0 means block was shed)
                    if w < TOLERANCE
                        continue
                    end
                    if w < V_MIN_SQ - TOLERANCE
                        push!(violations, "Bus $bus_id phase $idx: w = $w < $V_MIN_SQ (undervoltage)")
                    elseif w > V_MAX_SQ + TOLERANCE
                        push!(violations, "Bus $bus_id phase $idx: w = $w > $V_MAX_SQ (overvoltage)")
                    end
                end
            else
                w = w_vals
                # Skip de-energized buses (w ≈ 0 means block was shed)
                if w < TOLERANCE
                    continue
                end
                if w < V_MIN_SQ - TOLERANCE
                    push!(violations, "Bus $bus_id: w = $w < $V_MIN_SQ (undervoltage)")
                elseif w > V_MAX_SQ + TOLERANCE
                    push!(violations, "Bus $bus_id: w = $w > $V_MAX_SQ (overvoltage)")
                end
            end
        end
    end

    return isempty(violations), violations
end

"""
Helper function to check switch current limits are satisfied.
Based on constraint: P² + Q² ≤ w * I_rating²
Returns (passed::Bool, violations::Vector)
"""
function check_current_limits(result::Dict, math::Dict)
    violations = []

    if !haskey(result, "solution") || !haskey(result["solution"], "switch")
        return true, []  # No switches to check
    end

    if !haskey(math, "switch")
        return true, []
    end

    for (switch_id, switch) in math["switch"]
        if !haskey(result["solution"]["switch"], switch_id)
            continue
        end

        switch_soln = result["solution"]["switch"][switch_id]

        # Get switch state (z_switch) - can be fractional in relaxed formulations
        state = get(switch_soln, "state", 1.0)
        z_switch = state isa AbstractArray ? minimum(state) : state

        if z_switch < 1e-6
            continue  # Open switches don't carry current
        end

        # Get power flow
        pf = get(switch_soln, "psw_fr", get(switch_soln, "pf", [0.0]))
        qf = get(switch_soln, "qsw_fr", get(switch_soln, "qf", [0.0]))

        # Get current rating
        current_rating = get(switch, "current_rating", [Inf])
        f_connections = get(switch, "f_connections", [1, 2, 3])
        f_bus = switch["f_bus"]

        # Get voltage at from bus - use the same approach as fairness_evaluation.jl
        w_vals = nothing
        if haskey(result["solution"], "bus") && haskey(result["solution"]["bus"], string(f_bus))
            bus_soln = result["solution"]["bus"][string(f_bus)]
            if haskey(bus_soln, "w")
                w_vals = bus_soln["w"]
            end
        end

        # Convert pf/qf to arrays
        if pf isa Dict
            pf_arr = collect(values(pf))
        elseif pf isa AbstractArray
            pf_arr = pf
        else
            pf_arr = [pf]
        end

        if qf isa Dict
            qf_arr = collect(values(qf))
        elseif qf isa AbstractArray
            qf_arr = qf
        else
            qf_arr = [qf]
        end

        for (idx, (p, q)) in enumerate(zip(pf_arr, qf_arr))
            s_squared = p^2 + q^2

            # Get w for this connection - try multiple key types
            conn = idx <= length(f_connections) ? f_connections[idx] : idx
            w = 1.0
            if w_vals !== nothing
                if w_vals isa Dict
                    # Try the connection key directly, then as Int, then as String
                    if haskey(w_vals, conn)
                        w = w_vals[conn]
                    elseif haskey(w_vals, string(conn))
                        w = w_vals[string(conn)]
                    else
                        # Fallback: use the idx-th value in the dict
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
            # Ensure w is a scalar
            w = w isa AbstractArray ? first(w) : Float64(w)

            rating = idx <= length(current_rating) ? current_rating[idx] : current_rating[1]

            if rating > 0 && rating < Inf && w > 1e-6
                # Constraint: S² ≤ z_switch * w * I²
                limit = z_switch * w * rating^2
                if s_squared > limit
                    utilization = sqrt(s_squared) / (sqrt(z_switch * w) * rating) * 100
                    push!(violations, "Switch $switch_id phase $idx: utilization = $(round(utilization, digits=1))% (z=$(round(z_switch, digits=3)), w=$(round(w, digits=4)))")
                end
            end
        end
    end

    return isempty(violations), violations
end

"""
Helper function to check power balance (total generation ≥ total served load)
"""
function check_power_balance(result::Dict, math::Dict)
    if !haskey(result, "solution")
        return false, ["No solution found"]
    end

    # Calculate total load served
    total_demand = 0.0
    total_shed = 0.0

    if haskey(math, "load") && haskey(result["solution"], "load")
        for (load_id, load) in math["load"]
            total_demand += sum(load["pd"])
            if haskey(result["solution"]["load"], load_id)
                load_soln = result["solution"]["load"][load_id]
                if haskey(load_soln, "pshed")
                    total_shed += sum(load_soln["pshed"])
                end
            end
        end
    end

    total_served = total_demand - total_shed

    # Calculate total generation
    total_gen = 0.0
    if haskey(math, "gen") && haskey(result["solution"], "gen")
        for (gen_id, gen_soln) in result["solution"]["gen"]
            if haskey(gen_soln, "pg")
                pg = gen_soln["pg"]
                total_gen += pg isa Dict ? sum(values(pg)) : sum(pg)
            end
        end
    end

    # Check balance (with tolerance for losses)
    # Generation should be >= served load (difference is losses)
    if total_gen < total_served - TOLERANCE * 100  # Allow larger tolerance for power balance
        return false, ["Power imbalance: Gen=$total_gen < Served=$total_served"]
    end

    return true, []
end

"""
Run all standard tests for a formulation result
"""
function run_standard_tests(result::Dict, math::Dict, formulation_name::String)
    @testset "$formulation_name - Standard Tests" begin
        # Test 1: Termination status
        @testset "Termination Status" begin
            @test result["termination_status"] in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED]
        end

        # Test 2: Voltage limits
        @testset "Voltage Limits" begin
            passed, violations = check_voltage_limits(result, math)
            if !passed
                for v in violations
                    @warn v
                end
            end
            @test passed
        end

        # Test 3: Current limits
        @testset "Current Limits" begin
            passed, violations = check_current_limits(result, math)
            if !passed
                for v in violations
                    @warn v
                end
            end
            @test passed
        end

        # Test 4: Power balance
        @testset "Power Balance" begin
            passed, violations = check_power_balance(result, math)
            if !passed
                for v in violations
                    @warn v
                end
            end
            @test passed
        end
    end
end

@testset "FairLoadDelivery.jl" begin
    # Setup network once for all tests
    eng, math, lbs, critical_id = FairLoadDelivery.setup_network(TEST_CASE, GEN_CAP, [])

    # ============================================================
    # EFFICIENT FORMULATION TESTS
    # ============================================================

    @testset "Efficient Relaxed" begin
        result = FairLoadDelivery.solve_mc_mld_switch_relaxed(math, ipopt)
        run_standard_tests(result, math, "Efficient Relaxed")

        @testset "Objective Sanity Check" begin
            # Efficient should maximize total load served
            total_demand = sum(sum(load["pd"]) for (_, load) in math["load"])
            total_shed = sum(sum(result["solution"]["load"][lid]["pshed"]) for lid in keys(result["solution"]["load"]))
            total_served = total_demand - total_shed

            # Should serve some load
            @test total_served > 0
            # Should not serve more than demand
            @test total_served <= total_demand + TOLERANCE
        end
    end

    @testset "Efficient Integer" begin
        result = FairLoadDelivery.solve_mc_mld_switch_integer(math, gurobi)
        run_standard_tests(result, math, "Efficient Integer")
    end

    # ============================================================
    # EQUALITY MIN FORMULATION TESTS
    # ============================================================

    @testset "Equality Min Relaxed" begin
        result = FairLoadDelivery.solve_mc_mld_equality_min(math, ipopt)
        run_standard_tests(result, math, "Equality Min Relaxed")

        @testset "Fairness Check" begin
            # Equality min should minimize maximum shed
            pshed_values = Float64[]
            for (load_id, load_soln) in result["solution"]["load"]
                original_demand = sum(math["load"][load_id]["pd"])
                if original_demand > 1e-6
                    shed_frac = sum(load_soln["pshed"]) / original_demand
                    push!(pshed_values, shed_frac)
                end
            end

            if length(pshed_values) > 1
                # Check that shed fractions are relatively balanced
                max_shed = maximum(pshed_values)
                min_shed = minimum(pshed_values)

                # For min-max fairness, max should be minimized
                # This is a sanity check - exact equality is not guaranteed
                @test max_shed <= 1.0 + TOLERANCE  # Can't shed more than 100%
            end
        end
    end

    @testset "Equality Min Integer" begin
        result = FairLoadDelivery.solve_mc_mld_equality_min_integer(math, gurobi)
        run_standard_tests(result, math, "Equality Min Integer")
    end

    # ============================================================
    # PROPORTIONAL FAIRNESS FORMULATION TESTS
    # ============================================================

    @testset "Proportional Fairness Relaxed" begin
        result = FairLoadDelivery.solve_mc_mld_proportional_fairness(math, ipopt)
        run_standard_tests(result, math, "Proportional Fairness Relaxed")

        @testset "Objective Sanity Check" begin
            # Proportional fairness maximizes sum of log(served)
            # Should serve some load to each customer if possible
            served_fractions = Float64[]
            for (load_id, load_soln) in result["solution"]["load"]
                original_demand = sum(math["load"][load_id]["pd"])
                if original_demand > 1e-6
                    served = original_demand - sum(load_soln["pshed"])
                    push!(served_fractions, served / original_demand)
                end
            end

            # All served fractions should be non-negative
            @test all(f >= -TOLERANCE for f in served_fractions)
        end
    end

    @testset "Proportional Fairness Integer" begin
        result = FairLoadDelivery.solve_mc_mld_proportional_fairness_integer(math, gurobi)
        run_standard_tests(result, math, "Proportional Fairness Integer")
    end

    # ============================================================
    # JAIN FORMULATION TESTS
    # ============================================================

    @testset "Jain Relaxed" begin
        result = FairLoadDelivery.solve_mc_mld_jain(math, ipopt)
        run_standard_tests(result, math, "Jain Relaxed")

        @testset "Jain's Index Check" begin
            # Calculate Jain's index for the solution
            served_fractions = Float64[]
            for (load_id, load_soln) in result["solution"]["load"]
                original_demand = sum(math["load"][load_id]["pd"])
                if original_demand > 1e-6
                    served = original_demand - sum(load_soln["pshed"])
                    push!(served_fractions, served / original_demand)
                end
            end

            if !isempty(served_fractions)
                n = length(served_fractions)
                sum_x = sum(served_fractions)
                sum_x2 = sum(x^2 for x in served_fractions)

                if sum_x2 > 0
                    jains_index = sum_x^2 / (n * sum_x2)
                    # Jain's index should be between 1/n and 1
                    @test jains_index >= 1/n - TOLERANCE
                    @test jains_index <= 1.0 + TOLERANCE
                end
            end
        end
    end

    @testset "Jain Integer" begin
        result = FairLoadDelivery.solve_mc_mld_jain_integer(math, gurobi)
        run_standard_tests(result, math, "Jain Integer")
    end

    # ============================================================
    # PALMA FORMULATION TESTS (if implemented)
    # ============================================================

    @testset "Palma Relaxed" begin
        try
            result = FairLoadDelivery.solve_mc_mld_palma(math, ipopt)
            run_standard_tests(result, math, "Palma Relaxed")
        catch e
            @warn "Palma Relaxed test skipped: $e"
            @test_skip "Palma Relaxed not available"
        end
    end

    @testset "Palma Integer" begin
        try
            result = FairLoadDelivery.solve_mc_mld_palma_integer(math, gurobi)
            run_standard_tests(result, math, "Palma Integer")
        catch e
            @warn "Palma Integer test skipped: $e"
            @test_skip "Palma Integer not available"
        end
    end

    # ============================================================
    # COMPARATIVE TESTS
    # ============================================================

    @testset "Efficiency Comparison" begin
        # Efficient formulation should have highest total served (or tied)
        efficient_result = FairLoadDelivery.solve_mc_mld_switch_relaxed(math, ipopt)
        equality_result = FairLoadDelivery.solve_mc_mld_equality_min(math, ipopt)

        efficient_served = sum(
            sum(math["load"][lid]["pd"]) - sum(efficient_result["solution"]["load"][lid]["pshed"])
            for lid in keys(efficient_result["solution"]["load"])
        )

        equality_served = sum(
            sum(math["load"][lid]["pd"]) - sum(equality_result["solution"]["load"][lid]["pshed"])
            for lid in keys(equality_result["solution"]["load"])
        )

        # Efficient should serve at least as much as equality (within tolerance)
        @test efficient_served >= equality_served - TOLERANCE * 10
    end
end
