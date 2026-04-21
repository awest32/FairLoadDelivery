using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra, SparseArrays
using DataFrames
using CSV
using Dates
using Plots
using StatsPlots
using Statistics

include("validation_utils.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/load_shed_as_parameter.jl")


"""
    Efficient script to compare load shed results from the bilevel formulation
    across all cases and fairness functions.
"""
# ============================================================
# CONFIGURATION
# ============================================================
const CASES = ["motivation_c"]#["ieee123_aw_mod"]
const FAIR_FUNCS = ["efficiency",  "min_max","palma", "jain", "equality_min", "proportional"]#min_max throws error for motivation_c
const LS_PERCENT = 0.8 #20% load shed, 80% generation capacity
const ITERATIONS = 20 
const N_ROUNDS = 1
const N_BERNOULLI_SAMPLES = 2000#000
const SOURCE_PU = 1.03
const critical_buses = []# ["611c"]
# Save results
save_dir = "results/$(Dates.today())/bilevel_comparisons_single_period"
mkpath(save_dir)


# Solvers
ipopt_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
function extract_switch_block_states(relaxed_soln::Dict{String,Any})
    switch_states = Dict{Int64,Float64}()
    for (s_id, s_data) in relaxed_soln["switch"]
        switch_states[parse(Int, s_id)] = s_data["state"]
    end

    block_status = Dict{Int64,Float64}()
    for (b_id, b_data) in relaxed_soln["block"]
        block_status[parse(Int, b_id)] = b_data["status"]
    end
    return switch_states, block_status
end

# ============================================================
# STREAMLINED BILEVEL OPTIMIZATION (no plotting)
# ============================================================
function run_bilevel_relaxed(data::Dict{String, Any}, iterations::Int, fair_weights_init::Vector{Float64}, fair_func::String, critical_id::Vector{Int}=Int[])
    math_new = deepcopy(data)
    fair_weights = copy(fair_weights_init)  # Mutable copy for updates
    pshed_lower_level = Float64[]
    pshed_upper_level = Float64[]
    final_weight_ids = Int[]
    final_weights = Float64[]
    completed_iterations = 0
    last_status = MOI.OPTIMIZE_NOT_CALLED
    prev_weights = copy(fair_weights)
    prev_pshed = Float64[]
    max_delta_weights = NaN
    max_delta_pshed = NaN

    for k in 1:iterations
        # Solve lower-level problem and get sensitivities
        dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, _ = lower_level_soln(math_new, fair_weights, k)

        # Apply fairness function
        if fair_func == "proportional"
            pd = Float64[sum(math_new["load"][string(i)]["pd"]) for i in pshed_ids]
            pshed_new, fair_weight_vals, status = proportional_fairness_load_shed(dpshed, pshed_val, weight_vals, pd, critical_id, weight_ids)
        elseif fair_func == "efficiency"
            pshed_new, fair_weight_vals, status = efficient_load_shed(dpshed, pshed_val, weight_vals; critical_id, weight_ids)
        elseif fair_func == "min_max"
            pshed_new, fair_weight_vals, status = min_max_load_shed(dpshed, pshed_val, weight_vals, critical_id, weight_ids)
        elseif fair_func == "equality_min"
            pshed_new, fair_weight_vals, status = FairLoadDelivery.equality_min(dpshed, pshed_val, weight_vals, critical_id, weight_ids)
        elseif fair_func == "jain"
            pshed_new, fair_weight_vals, status = jains_fairness_index(dpshed, pshed_val, weight_vals, critical_id, weight_ids)
        elseif fair_func == "palma"
            pd = Float64[]
            for i in pshed_ids
                push!(pd, sum(math_new["load"][string(i)]["pd"]))
            end
            pshed_new, fair_weight_vals, status = lin_palma_reformulated(dpshed, pshed_val, weight_vals, pd, critical_ids, weight_ids)
        else
            error("Unknown fairness function: $fair_func")
        end

        # If upper-level fairness problem is infeasible, stop bilevel iteration and keep last feasible weights
        last_status = status
        @info "[$fair_func] Iteration $k: upper-level status = $status (type: $(typeof(status)))"
        if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.TIME_LIMIT]
            @warn "[$fair_func] Iteration $k/$iterations: upper-level infeasible (status=$status) — stopping bilevel with last feasible weights"
            break
        end
        completed_iterations = k

        # Track convergence: max absolute change in weights and pshed
        max_delta_weights = maximum(abs.(fair_weight_vals .- prev_weights))
        if !isempty(prev_pshed)
            max_delta_pshed = maximum(abs.(pshed_new .- prev_pshed))
        end
        prev_weights = copy(fair_weight_vals)
        prev_pshed = copy(pshed_new)

        # Update weights in math dictionary
        for (i, w) in zip(weight_ids, fair_weight_vals)
            math_new["load"][string(i)]["weight"] = w
        end

        # Update fair_weights for next iteration
        fair_weights = fair_weight_vals
        final_weight_ids = weight_ids
        final_weights = fair_weight_vals

        push!(pshed_lower_level, sum(pshed_val))
        push!(pshed_upper_level, sum(pshed_new))
    end

    weights_converged = !isnan(max_delta_weights) && max_delta_weights <= TRUST_RADIUS
    bilevel_summary = Dict(
        "completed_iterations" => completed_iterations,
        "total_iterations" => iterations,
        "last_status" => string(last_status),
        "early_stop" => completed_iterations < iterations,
        "max_delta_weights" => max_delta_weights,
        "max_delta_pshed" => max_delta_pshed,
        "weights_converged" => weights_converged
    )
    @info "[$fair_func] Bilevel finished: $completed_iterations/$iterations iterations, last status=$last_status, Δw_max=$(round(max_delta_weights, digits=6)), weights_converged=$weights_converged"
    return math_new, pshed_lower_level, pshed_upper_level, final_weight_ids, final_weights, bilevel_summary
end

# ============================================================
# RANDOM ROUNDING AND FINAL SOLUTION
# ============================================================
function run_random_rounding(math_relaxed::Dict, n_rounds::Int, n_samples::Int, ipopt; fair_func::String="", case::String="", mld_relaxed::Union{Dict,Nothing}=nothing)
    # Solve implicit diff to get switch/block states
    mld_implicit = solve_mc_mld_shed_implicit_diff(math_relaxed, ipopt; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])

    imp_model = instantiate_mc_model(
        math_relaxed,
        LinDist3FlowPowerModel,
        build_mc_mld_shedding_implicit_diff;
        ref_extensions=[FairLoadDelivery.ref_add_load_blocks!]
    )
    ref = imp_model.ref[:it][:pmd][:nw][0]

    switch_states, block_status = extract_switch_block_states(mld_implicit["solution"])

    # Storage for each round
   math_radial = Vector{Dict{String, Any}}()
    math_out = Vector{Dict{String, Any}}()
    mld_results = Vector{Dict{String, Any}}()
    math_out_ac = Vector{Dict{String, Any}}()
    ac_results = Vector{Dict{String, Any}}()
    ac_tested = Int[]
    round_tested = Int[]
    stage = "random_rounding"

    for r in 1:N_ROUNDS
        rng = 100 * r
        bernoulli_samples = generate_bernoulli_samples(switch_states, N_BERNOULLI_SAMPLES, rng);

        index, switch_states_radial, block_ids, block_status_radial, load_ids, load_status =
            radiality_check(ref, switch_states, block_status, bernoulli_samples)

        if index === nothing
            if r == N_ROUNDS && isempty(mld_results)
                @error "[$case/$fair_func]: All $N_ROUNDS rounds failed at RADIAL FEASIBILITY — no Bernoulli sample produced a feasible radial topology"
                return math_out, mld_results, ref
            else
                @warn "[$case/$fair_func]: Round $r failed at: RADIAL FEASIBILITY — skipping to next round"
                continue
            end
        end

        # Apply rounded states (deepcopy so successive rounds don't corrupt each other)
        math_rounded = update_network(deepcopy(math_relaxed), switch_states_radial, ref);
        push!(math_radial, math_rounded)

        # Solve rounded MLD
        if !isempty(math_rounded)
            mld_rounded = FairLoadDelivery.solve_mc_mld_shed_random_round_integer(math_rounded, gurobi);
            if mld_rounded["termination_status"] in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
                push!(math_out, math_rounded)
                push!(mld_results, mld_rounded)
            else
                @warn "[$case/$fair_func]: Round $r failed at: ROUNDED MLD SOLVE — termination status: $(mld_rounded["termination_status"])"
            end
        else
            @warn "Round $r failed: no valid radial topology found, skipping rounded MLD solve and ACPF."
        end
    end
    return math_out, mld_results, ref
end

function find_best_mld_solution(mlds::Vector{Dict{String, Any}})
    best_obj = -Inf
    best_idx = 0
    for (idx, mld) in enumerate(mlds)
        if mld["objective"] > best_obj
            best_obj = mld["objective"]
            best_idx = idx
        end
    end
    return best_idx, best_idx > 0 ? mlds[best_idx] : nothing
end

function check_binary_solution(mld::Dict)
    """Check that switch states and block status in final MLD solution are binary (0.0 or 1.0)."""
    if mld === nothing || !haskey(mld, "solution")
        return true, []
    end

    violations = String[]

    # Check switch states
    if haskey(mld["solution"], "switch")
        for (s_id, s_data) in mld["solution"]["switch"]
            state = s_data["state"]
            if !(isapprox(state, 0.0, atol=1e-6) || isapprox(state, 1.0, atol=1e-6))
                push!(violations, "Switch $s_id: state=$state")
            end
        end
    end

    # Check block status
    if haskey(mld["solution"], "block")
        for (b_id, b_data) in mld["solution"]["block"]
            status = b_data["status"]
            if !(isapprox(status, 0.0, atol=1e-6) || isapprox(status, 1.0, atol=1e-6))
                push!(violations, "Block $b_id: status=$status")
            end
        end
    end

    return isempty(violations), violations
end

function run_acpf_on_rounded_solution(math_rounded::Dict, ipopt_solver, ref::Dict{Symbol,Any}, mld_solution::Dict{String,Any})
    ac_checks = Dict{String, Any}()
    ac_summary = Dict{String, Any}()

    # Use ac_network_update to de-energize disconnected loads, shunts, branches
    math_ac = ac_network_update(math_rounded, ref; mld_solution=mld_solution)

    # Run AC power flow
    println("  Running AC power flow (IVRUPowerModel)...")
    ac_result = PowerModelsDistribution.solve_mc_pf(math_ac, IVRUPowerModel, ipopt_solver)

    ac_term = ac_result["termination_status"]
    ac_converged = (ac_term == MOI.OPTIMAL || ac_term == MOI.LOCALLY_SOLVED || ac_term == MOI.ALMOST_LOCALLY_SOLVED)

    ac_checks["ac_convergence"] = Dict("passed" => ac_converged, "details" => ["Status: $ac_term"])

    # Post-solve voltage feasibility check on energized buses
    voltage_violations = String[]
    if ac_converged && haskey(ac_result, "solution") && haskey(ac_result["solution"], "bus")
        for (bid, bus_sol) in ac_result["solution"]["bus"]
            bus_data = math_ac["bus"][bid]
            # Skip de-energized buses
            if get(bus_data, "status", 1) == 0 || get(bus_data, "bus_type", 1) == 4
                continue
            end
            if haskey(bus_sol, "vm")
                vmin = bus_data["vmin"]
                vmax = bus_data["vmax"]
                for (idx, vm) in enumerate(bus_sol["vm"])
                    if vm < vmin[idx] - 1e-4 || vm > vmax[idx] + 1e-4
                        push!(voltage_violations, "Bus $(bus_data["name"]) phase $idx: vm=$(round(vm, digits=4)) outside [$(vmin[idx]), $(vmax[idx])]")
                    end
                end
            end
        end
    end

    voltage_feasible = isempty(voltage_violations)
    if !voltage_feasible
        @warn "ACPF voltage violations ($(length(voltage_violations))): $(join(voltage_violations[1:min(5, length(voltage_violations))], "; "))"
    end

    ac_checks["voltage_feasibility"] = Dict("passed" => voltage_feasible, "details" => voltage_violations)
    ac_summary["converged"] = ac_converged
    ac_summary["voltage_feasible"] = voltage_feasible
    ac_summary["termination_status"] = string(ac_term)
    return ac_result, math_ac, ac_checks, ac_summary
end

function extract_load_shed(mld::Dict)
    if mld === nothing || !haskey(mld, "solution") || !haskey(mld["solution"], "load")
       @error "MLD solution does not contain load data"
    end

    total_pshed = 0.0
    total_pd_served = 0.0
    for (load_id, load_data) in mld["solution"]["load"]
        if haskey(load_data, "pshed")
            total_pshed += sum(load_data["pshed"])
        end
        if haskey(load_data, "pd")
            total_pd_served += sum(load_data["pd"])
        end
    end
    return total_pshed, total_pd_served
end

function extract_per_load_data(mld::Dict, math::Dict; original_math::Union{Dict,Nothing}=nothing)
    """Extract per-bus shed and served percentages (P and Q), one entry per bus.
    Original demand is taken from `original_math` when provided, since `math` may be
    a post-rounding copy where de-energized loads have pd/qd zeroed out.
    Shed = original_demand - served. Loads not in solution are fully shed.
    `something(a, b)` is a Julia built-in that returns the first non-nothing argument."""

    # something(a, b): Julia built-in — returns `a` if not nothing, else `b`
    orig_math = something(original_math, math)

    # Group loads by bus
    bus_loads = Dict{Int, Vector{Int}}()  # bus_id => [load_ids...]
    for (lid_str, load_data) in math["load"]
        lid = parse(Int, lid_str)
        bus_id = load_data["load_bus"]
        if !haskey(bus_loads, bus_id)
            bus_loads[bus_id] = Int[]
        end
        push!(bus_loads[bus_id], lid)
    end

    bus_ids = sort(collect(keys(bus_loads)))
    pshed_pct = Float64[]
    pd_served_pct = Float64[]
    qshed_pct = Float64[]
    qd_served_pct = Float64[]
    pd_served_kw = Float64[]
    pd_demand_kw = Float64[]

    sol_load = get(get(mld, "solution", Dict()), "load", Dict{String,Any}())

    for bus_id in bus_ids
        # Aggregate original demand and served across all loads at this bus
        pd_total = 0.0
        qd_total = 0.0
        pd_served = 0.0
        qd_served = 0.0

        for lid in bus_loads[bus_id]
            lid_str = string(lid)
            # Original demand from pre-rounding math
            orig_load = orig_math["load"][lid_str]
            pd_total += sum(orig_load["pd"])
            qd_total += sum(orig_load["qd"])

            # Served from solution; missing loads are fully shed (served = 0)
            if haskey(sol_load, lid_str)
                load_sol = sol_load[lid_str]
                if haskey(load_sol, "pd")
                    pd_s = load_sol["pd"]
                    pd_served += isa(pd_s, AbstractArray) ? sum(pd_s) : pd_s
                end
                if haskey(load_sol, "qd")
                    qd_s = load_sol["qd"]
                    qd_served += isa(qd_s, AbstractArray) ? sum(qd_s) : qd_s
                end
            end
        end

        pshed = pd_total - pd_served
        qshed = qd_total - qd_served

        push!(pshed_pct, pd_total > 0 ? (pshed / pd_total) * 100 : 0.0)
        push!(pd_served_pct, pd_total > 0 ? (pd_served / pd_total) * 100 : 0.0)
        push!(qshed_pct, qd_total > 0 ? (qshed / qd_total) * 100 : 0.0)
        push!(qd_served_pct, qd_total > 0 ? (qd_served / qd_total) * 100 : 0.0)
        push!(pd_served_kw, pd_served)
        push!(pd_demand_kw, pd_total)
    end

    return bus_ids, pshed_pct, pd_served_pct, qshed_pct, qd_served_pct, pd_served_kw, pd_demand_kw
end

function extract_original_load(math::Dict)
    """Extract original pd and qd values from math dictionary, sorted by load ID."""
    load_ids = sort(parse.(Int, collect(keys(math["load"]))))
    pd_original = Float64[]
    qd_original = Float64[]
    load_names = String[]

    for lid in load_ids
        load_data = math["load"][string(lid)]
        push!(pd_original, sum(load_data["pd"]))
        push!(qd_original, sum(load_data["qd"]))
        push!(load_names, load_data["name"])
    end

    return load_ids, load_names, pd_original, qd_original
end

function plot_original_load_bars(case::String, math::Dict, save_dir::String)
    """Create bar plots for original pd and qd per load."""
    load_ids, load_names, pd_original, qd_original = extract_original_load(math)
    x_positions = collect(1:length(load_ids))

    # Active power plot
    p_pd = bar(
        x_positions, pd_original,
        xlabel = "Load",
        ylabel = "Active Power (kW)",
        title = "Original Active Power Demand: $case",
        legend = false,
        xticks = (x_positions, load_names),
        xrotation = 45,
        size = (1200, 600),
        left_margin = 10Plots.mm,
        bottom_margin = 15Plots.mm,
        top_margin = 5Plots.mm,
        color = :blue
    )
    savefig(p_pd, joinpath(save_dir, "original_pd_$case.svg"))

    # Reactive power plot
    p_qd = bar(
        x_positions, qd_original,
        xlabel = "Load",
        ylabel = "Reactive Power (kVAR)",
        title = "Original Reactive Power Demand: $case",
        legend = false,
        xticks = (x_positions, load_names),
        xrotation = 45,
        size = (1200, 600),
        left_margin = 10Plots.mm,
        bottom_margin = 15Plots.mm,
        top_margin = 5Plots.mm,
        color = :red
    )
    savefig(p_qd, joinpath(save_dir, "original_qd_$case.svg"))

    # Combined P and Q plot
    p_combined = groupedbar(
        load_names, hcat(pd_original, qd_original),
        xlabel = "Load",
        ylabel = "Power",
        title = "Original Demand (P and Q): $case",
        label = ["Active Power (kW)" "Reactive Power (kVAR)"],
        legend = :outertopright,
        xrotation = 45,
        size = (1200, 600),
        left_margin = 10Plots.mm,
        bottom_margin = 15Plots.mm,
        top_margin = 5Plots.mm,
        color = [:blue :red]
    )
    savefig(p_combined, joinpath(save_dir, "original_demand_$case.svg"))

    return p_pd, p_qd, p_combined
end

function plot_original_network_load(case::String, math::Dict, save_dir::String)
    """Create network visualization showing original load distribution."""
    # Create a mock solution with all loads fully served (no shedding)
    mock_solution = Dict{String, Any}()

    # Load data - all served, no shed
    mock_solution["load"] = Dict{String, Any}()
    for (lid, load_data) in math["load"]
        mock_solution["load"][lid] = Dict(
            "pd" => load_data["pd"],
            "qd" => load_data["qd"],
            "pshed" => zeros(length(load_data["pd"])),
            "qshed" => zeros(length(load_data["qd"])),
            "status" => 1.0
        )
    end

    # Bus data - all on with nominal voltage
    mock_solution["bus"] = Dict{String, Any}()
    for (bid, bus_data) in math["bus"]
        n_phases = length(bus_data["terminals"])
        mock_solution["bus"][bid] = Dict(
            "w" => ones(n_phases),
            "status" => 1.0
        )
    end

    # Switch data - all closed
    mock_solution["switch"] = Dict{String, Any}()
    if haskey(math, "switch")
        for (sid, switch_data) in math["switch"]
            n_phases = length(switch_data["f_connections"])
            mock_solution["switch"][sid] = Dict(
                "state" => 1.0,
                "pf" => zeros(n_phases),
                "qf" => zeros(n_phases)
            )
        end
    end

    # Block data - all on
    mock_solution["block"] = Dict{String, Any}()
    if haskey(math, "block")
        for (bid, _) in math["block"]
            mock_solution["block"][bid] = Dict("status" => 1.0)
        end
    end

    # Plot using existing function
    plot_filename = joinpath(save_dir, "network_original_$case.svg")
    FairLoadDelivery.plot_network_load_shed(mock_solution, math; output_file=plot_filename)
end

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
# FAIR_FUNC_COLORS, FAIR_FUNC_LABELS, FAIR_FUNC_MARKERS now imported from FairLoadDelivery

function create_grouped_bar_chart(case::String, per_load_data::Dict, data_type::Symbol, save_dir::String)
    """
    Create grouped bar chart for load shed or load served.
    data_type: :pshed or :pd_served
    """
    # Get union of all load IDs and build ID-to-name mapping
    all_load_ids = Set{Int}()
    id_to_name = Dict{Int,String}()
    for (_, data) in per_load_data
        if !isempty(data[:load_ids])
            union!(all_load_ids, data[:load_ids])
            for (lid, lname) in zip(data[:load_ids], data[:load_names])
                id_to_name[lid] = lname
            end
        end
    end

    if isempty(all_load_ids)
        @warn "No load data available for $case"
        return nothing
    end

    load_ids = sort(collect(all_load_ids))
    load_names = [id_to_name[lid] for lid in load_ids]

    # Count functions with actual data
    n_funcs_with_data = count(ff -> haskey(per_load_data, ff) && !isempty(per_load_data[ff][:load_ids]), FAIR_FUNCS)
    if n_funcs_with_data == 0
        @warn "No valid data for $case"
        return nothing
    end

    bar_width = 0.8 / n_funcs_with_data
    x_offset = bar_width

    if data_type == :pshed
        title = "Active Power Load Shed per Bus: $case"
        ylabel = "Load Shed (%)"
        filename = "pshed_per_bus_$case.svg"
    elseif data_type == :pd_served
        title = "Active Power Load Served per Bus: $case"
        ylabel = "Load Served (%)"
        filename = "pd_served_per_bus_$case.svg"
    elseif data_type == :qshed
        title = "Reactive Power Load Shed per Bus: $case"
        ylabel = "Load Shed (%)"
        filename = "qshed_per_bus_$case.svg"
    elseif data_type == :qd_served
        title = "Reactive Power Load Served per Bus: $case"
        ylabel = "Load Served (%)"
        filename = "qd_served_per_bus_$case.svg"
    else
        error("Unknown data_type: $data_type")
    end

    x_positions = collect(1:length(load_ids))

    p = plot(
        xlabel = "Bus",
        ylabel = ylabel,
        title = title,
        legend = :outertopright,
        xticks = (x_positions, load_names),
        xrotation = 45,
        size = (1200, 600),
        left_margin = 10Plots.mm,
        bottom_margin = 15Plots.mm,
        top_margin = 5Plots.mm
    )

    # Calculate offsets to center the bars
    offsets = collect(range(-(n_funcs_with_data-1)/2 * x_offset, (n_funcs_with_data-1)/2 * x_offset, length=n_funcs_with_data))

    func_idx = 0
    for fair_func in FAIR_FUNCS
        if !haskey(per_load_data, fair_func)
            continue
        end

        data = per_load_data[fair_func]
        func_load_ids = data[:load_ids]
        func_values = data[data_type]

        if isempty(func_load_ids)
            continue
        end

        func_idx += 1

        # Create a mapping from load_id to value for this function
        id_to_value = Dict(zip(func_load_ids, func_values))

        # Build values array aligned with the common load_ids (0 for missing)
        aligned_values = [get(id_to_value, lid, 0.0) for lid in load_ids]

        bar!(p, x_positions .+ offsets[func_idx], aligned_values,
            bar_width = bar_width * 0.9,
            label = FAIR_FUNC_LABELS[fair_func],
            color = FAIR_FUNC_COLORS[fair_func]
        )
    end

    savefig(p, joinpath(save_dir, filename))
    return p
end

function create_shed_distribution_plot(case::String, per_load_data::Dict, power_type::Symbol, save_dir::String)
    """
    Create scatter plot showing shed distribution by fairness function.
    power_type: :pshed for active power, :qshed for reactive power
    """
    # Get union of all load IDs and build ID-to-name mapping
    all_load_ids = Set{Int}()
    id_to_name = Dict{Int,String}()
    for (_, data) in per_load_data
        if !isempty(data[:load_ids])
            union!(all_load_ids, data[:load_ids])
            for (lid, lname) in zip(data[:load_ids], data[:load_names])
                id_to_name[lid] = lname
            end
        end
    end

    if isempty(all_load_ids)
        return nothing
    end

    load_ids = sort(collect(all_load_ids))
    load_names = [id_to_name[lid] for lid in load_ids]
    x_positions = collect(1:length(load_ids))

    if power_type == :pshed
        ylabel = "Load Shed (%)"
        title = "Active Power Shed Distribution: $case"
        filename = "pshed_distribution_$case.svg"
    elseif power_type == :qshed
        ylabel = "Load Shed (%)"
        title = "Reactive Power Shed Distribution: $case"
        filename = "qshed_distribution_$case.svg"
    else
        error("Unknown power_type: $power_type")
    end

    p = plot(
        xlabel = "Load",
        ylabel = ylabel,
        title = title,
        legend = :outertopright,
        xticks = (x_positions, load_names),
        xrotation = 45,
        size = (1200, 600),
        left_margin = 10Plots.mm,
        bottom_margin = 15Plots.mm,
        top_margin = 5Plots.mm
    )

    for fair_func in FAIR_FUNCS
        if !haskey(per_load_data, fair_func)
            continue
        end

        data = per_load_data[fair_func]
        if isempty(data[:load_ids])
            continue
        end

        id_to_value = Dict(zip(data[:load_ids], data[power_type]))
        aligned_values = [haskey(id_to_value, lid) ? id_to_value[lid] : error("Load ID $lid not found in id_to_value") for lid in load_ids]

        scatter!(p, x_positions, aligned_values,
            label = FAIR_FUNC_LABELS[fair_func],
            marker = FAIR_FUNC_MARKERS[fair_func],
            markersize = 8,
            color = FAIR_FUNC_COLORS[fair_func]
        )
    end

    # Add horizontal line for equality_min max shed (fairness target)
    if haskey(per_load_data, "equality_min") && !isempty(per_load_data["equality_min"][power_type])
        max_eq_shed = maximum(per_load_data["equality_min"][power_type])
        hline!(p, [max_eq_shed], label="EqMin Max Shed", linestyle=:dash, color=:orange, linewidth=2)
    end

    savefig(p, joinpath(save_dir, filename))
    return p
end

# ============================================================
# VOLTAGE PER PHASE COMPARISON
# ============================================================
## Show all buses in voltage comparison (no filtering)

# extract_voltage_by_bus_name now imported from FairLoadDelivery

# plot_voltage_per_phase_comparison replaced by FairLoadDelivery.plot_voltage_per_bus_comparison

# ============================================================
# MAIN COMPARISON LOOP
# ============================================================
function get_total_demand(math::Dict)
    total_pd = 0.0
    for (_, load) in math["load"]
        total_pd += sum(load["pd"])
    end
    return total_pd
end

function run_comparison()
    # Results storage
    results = DataFrame(
        case = String[],
        fair_func = String[],
        total_demand = Float64[],
        final_pshed = Float64[],
        final_pd_served = Float64[],
        pct_shed = Float64[],
        pct_served = Float64[],
        relaxed_pshed_final = Float64[],
        objective = Float64[],
        bilevel_iters_completed = Int[],
        bilevel_iters_total = Int[],
        bilevel_last_status = String[],
        bilevel_early_stop = Bool[],
        max_delta_weights = Float64[],
        max_delta_pshed = Float64[],
        weights_converged = Bool[]
    )

    # Per-load data storage: case => fair_func => {load_ids, pshed, pd_served}
    per_load_results = Dict{String, Dict{String, Dict{Symbol, Vector}}}()

    # Final weights storage: case => fair_func => {weight_ids, weights, bilevel_summary}
    final_weights_results = Dict{String, Dict{String, Dict{Symbol, Any}}}()

    # Solutions storage for plotting: case => fair_func => {mld, math}
    solutions_for_plotting = Dict{String, Dict{String, Dict{Symbol, Any}}}()

    # Track failed combinations
    failed_combinations = Vector{Tuple{String, String, String}}()  # (case, fair_func, reason)

    println("=" ^ 60)
    println("LOAD SHED COMPARISON ACROSS CASES AND FAIRNESS FUNCTIONS")
    println("=" ^ 60)

    for case in CASES
        println("\n>>> Processing case: $case")

        # Setup network
        eng, math, lbs, critical_id = FairLoadDelivery.setup_network("ieee_13_aw_edit/$case.dss", LS_PERCENT, SOURCE_PU, critical_buses)
        sorted_load_ids = sort(parse.(Int, collect(keys(math["load"]))))
        fair_weights = Float64[math["load"][string(i)]["weight"] for i in sorted_load_ids]
        total_demand = get_total_demand(math)

        println("    Total demand: $(round(total_demand, digits=4))")

        # Plot original load
        plot_original_load_bars(case, math, save_dir)
        plot_original_network_load(case, math, save_dir)
        println("    Saved original demand plots for $case")

        # Initialize per-load storage for this case
        per_load_results[case] = Dict{String, Dict{Symbol, Vector}}()
        final_weights_results[case] = Dict{String, Dict{Symbol, Any}}()
        solutions_for_plotting[case] = Dict{String, Dict{Symbol, Any}}()

        for fair_func in FAIR_FUNCS
            print("  $fair_func: ")

            # Run bilevel relaxation
            math_relaxed, pshed_lower, pshed_upper, weight_ids, final_wts, bilevel_summary = run_bilevel_relaxed(
                math, ITERATIONS, fair_weights, fair_func, critical_id
            )

            # Skip this fairness function if bilevel failed on the first iteration
            if bilevel_summary["completed_iterations"] == 0
                @warn "[$case/$fair_func] FAILED — bilevel infeasible on first iteration (status=$(bilevel_summary["last_status"])). Skipping."
                push!(failed_combinations, (case, fair_func, "Bilevel infeasible on first iteration"))
                continue
            end

            # Store final weights and bilevel summary
            final_weights_results[case][fair_func] = Dict(
                :weight_ids => weight_ids,
                :weights => final_wts,
                :bilevel_summary => bilevel_summary
            )

            # Run random rounding
            math_out, mld_results, rr_ref = run_random_rounding(
                math_relaxed, N_ROUNDS, N_BERNOULLI_SAMPLES, ipopt_solver;
                fair_func=fair_func, case=case
            )

            if isempty(mld_results)
                @warn "[$case/$fair_func] FAILED — no feasible solution after random rounding (all $N_ROUNDS rounds failed). Check warnings above for failure stage (RADIAL FEASIBILITY or ROUNDED MLD SOLVE)."
                push!(failed_combinations, (case, fair_func, "No feasible solution after random rounding"))
                continue
            end

            # Find best solution
            best_idx, best_mld = find_best_mld_solution(mld_results)
            
            # Check binary values in final solution
            binary_ok, binary_violations = check_binary_solution(best_mld)
            if !binary_ok
                @warn "$case/$fair_func: Non-binary values in final MLD solution: $(join(binary_violations[1:min(3,length(binary_violations))], "; "))"
            end

            # Conduct ACPF on best solution to get accurate load shed values and voltage data for plotting


            final_pshed, final_pd_served = extract_load_shed(best_mld)
            @info "    Case=$case, FairFunc=$fair_func => Final shed: $(round(final_pshed, digits=4)), Served: $(round(final_pd_served, digits=4))"
            # Extract per-bus data (one entry per bus, since all phases have same shed %)
            bus_ids, pshed_per_bus, pd_per_bus, qshed_per_bus, qd_per_bus, pd_served_kw, pd_demand_kw = extract_per_load_data(best_mld, math_out[best_idx])
            bus_name_map = build_bus_name_maps(math_out[best_idx])
            bus_names = [get(bus_name_map, bid, "bus_$bid") for bid in bus_ids]
            per_load_results[case][fair_func] = Dict(
                :load_ids => bus_ids,
                :load_names => bus_names,
                :pshed => pshed_per_bus,
                :pd_served => pd_per_bus,
                :qshed => qshed_per_bus,
                :qd_served => qd_per_bus,
                :pd_served_kw => pd_served_kw,
                :pd_demand_kw => pd_demand_kw
            )

        
            acpf, math_ac, ac_checks, ac_summary = run_acpf_on_rounded_solution(math_out[best_idx], ipopt_solver, rr_ref, best_mld)

            # ── Voltage troubleshooting ──────────────────────────────
            let diag_file = joinpath(save_dir, "voltage_debug_$(case)_$(fair_func).txt")
                open(diag_file, "w") do io
                    println(io, "Voltage Diagnostic: $case / $fair_func")
                    println(io, "="^60)

                    mld_sol = best_mld["solution"]
                    acpf_sol = acpf["solution"]
                    math_mld = math_out[best_idx]

                    # Source bus identification
                    source_gen_id = nothing
                    for (gid, gen) in math_mld["gen"]
                        if gen["source_id"] == "voltage_source.source"
                            source_gen_id = gid
                            source_bus = gen["gen_bus"]
                            println(io, "\nSource generator: gen $gid, bus $source_bus")
                            println(io, "  vg (math_mld): $(gen["vg"])")
                            if haskey(math_ac, "gen") && haskey(math_ac["gen"], gid)
                                println(io, "  vg (math_ac):  $(math_ac["gen"][gid]["vg"])")
                            end
                            break
                        end
                    end

                    # Per-bus voltage comparison
                    println(io, "\n\nPer-Bus Voltage Comparison (MLD w→√w vs ACPF vm)")
                    println(io, "-"^80)
                    println(io, lpad("Bus", 6), "  ", rpad("Name", 12),
                            "  ", lpad("MLD_V1", 8), lpad("MLD_V2", 8), lpad("MLD_V3", 8),
                            "  ", lpad("AC_V1", 8), lpad("AC_V2", 8), lpad("AC_V3", 8),
                            "  ", lpad("ΔV1", 8), lpad("ΔV2", 8), lpad("ΔV3", 8))

                    bus_ids = sort(parse.(Int, collect(keys(math_mld["bus"]))))
                    for bid in bus_ids
                        bid_str = string(bid)
                        bus = math_mld["bus"][bid_str]
                        bus_name = get(bus, "name", "?")

                        # MLD voltages: w → sqrt(w)
                        mld_v = [0.0, 0.0, 0.0]
                        if haskey(mld_sol, "bus") && haskey(mld_sol["bus"], bid_str)
                            bs = mld_sol["bus"][bid_str]
                            if haskey(bs, "w")
                                for (idx, c) in enumerate(bus["terminals"])
                                    w = bs["w"][idx]
                                    mld_v[c] = w >= 0 ? sqrt(w) : 0.0
                                end
                            elseif haskey(bs, "vm")
                                for (idx, c) in enumerate(bus["terminals"])
                                    mld_v[c] = bs["vm"][idx]
                                end
                            end
                        end

                        # ACPF voltages: IVR gives vr/vi, compute vm = sqrt(vr²+vi²)
                        ac_v = [0.0, 0.0, 0.0]
                        if haskey(acpf_sol, "bus") && haskey(acpf_sol["bus"], bid_str)
                            bs = acpf_sol["bus"][bid_str]
                            if haskey(bs, "vr") && haskey(bs, "vi")
                                for (idx, c) in enumerate(bus["terminals"])
                                    ac_v[c] = sqrt(bs["vr"][idx]^2 + bs["vi"][idx]^2)
                                end
                            elseif haskey(bs, "vm")
                                for (idx, c) in enumerate(bus["terminals"])
                                    ac_v[c] = bs["vm"][idx]
                                end
                            elseif haskey(bs, "w")
                                for (idx, c) in enumerate(bus["terminals"])
                                    w = bs["w"][idx]
                                    ac_v[c] = w >= 0 ? sqrt(w) : 0.0
                                end
                            end
                        end

                        delta = ac_v .- mld_v
                        println(io, lpad(bid_str, 6), "  ", rpad(bus_name, 12),
                                "  ", join([lpad(round(v, digits=4), 8) for v in mld_v]),
                                "  ", join([lpad(round(v, digits=4), 8) for v in ac_v]),
                                "  ", join([lpad(round(d, digits=4), 8) for d in delta]))
                    end

                    # Load setpoint comparison
                    println(io, "\n\nLoad Setpoint Comparison (MLD pd vs math_ac pd)")
                    println(io, "-"^60)
                    if haskey(math_mld, "load")
                        load_ids = sort(parse.(Int, collect(keys(math_mld["load"]))))
                        for lid in load_ids
                            lid_str = string(lid)
                            mld_pd = haskey(mld_sol, "load") && haskey(mld_sol["load"], lid_str) ? get(mld_sol["load"][lid_str], "pd", []) : []
                            ac_pd = haskey(math_ac, "load") && haskey(math_ac["load"], lid_str) ? math_ac["load"][lid_str]["pd"] : []
                            println(io, "  Load $lid_str: MLD pd=$(round.(mld_pd, digits=6))  AC pd=$(round.(ac_pd, digits=6))")
                        end
                    end

                    println(io, "\n\nMLD termination: $(best_mld["termination_status"])")
                    println(io, "ACPF termination: $(acpf["termination_status"])")
                end
                println("    Saved voltage diagnostics to $diag_file")
            end
            # ─────────────────────────────────────────────────────────

            # Store solution for plotting
            solutions_for_plotting[case][fair_func] = Dict(
                :mld => best_mld,
                :mld_math => math_out[best_idx],
                :acpf => acpf,
                :acpf_math => math_ac,
                :checks => ac_checks,
                :summary => ac_summary
            )

            # Calculate percentages
            pct_shed = (final_pshed / total_demand) * 100
            pct_served = (final_pd_served / total_demand) * 100

            println("shed=$(round(pct_shed, digits=2))%, served=$(round(pct_served, digits=2))%")
            
            push!(results, (
                case,
                fair_func,
                total_demand,
                final_pshed,
                final_pd_served,
                pct_shed,
                pct_served,
                pshed_upper[end],
                best_mld["objective"],
                bilevel_summary["completed_iterations"],
                bilevel_summary["total_iterations"],
                bilevel_summary["last_status"],
                bilevel_summary["early_stop"],
                bilevel_summary["max_delta_weights"],
                bilevel_summary["max_delta_pshed"],
                bilevel_summary["weights_converged"]
            ))
        end
    end

    return results, per_load_results, final_weights_results, solutions_for_plotting, failed_combinations
end


# ============================================================
# RUN AND SAVE RESULTS
# ============================================================
println("\nStarting comparison at $(now())...\n")

results, per_load_results, final_weights_results, solutions_for_plotting, failed_combinations = run_comparison()

# Print failed combinations
if !isempty(failed_combinations)
    println("\n" * "=" ^ 60)
    println("FAILED COMBINATIONS")
    println("=" ^ 60)
    for (case, fair_func, reason) in failed_combinations
        println("  $case / $fair_func: $reason")
    end
end

# Print summary table
println("\n" * "=" ^ 60)
println("SUMMARY TABLE")
println("=" ^ 60)
println(results)

# Pivot table for easier comparison
println("\n" * "=" ^ 60)
println("LOAD SHED BY CASE AND FAIRNESS FUNCTION")
println("=" ^ 60)

for case in CASES
    println("\n$case:")
    case_results = filter(row -> row.case == case, results)
    if !isempty(case_results)
        println("  Total demand: $(round(first(case_results.total_demand), digits=4))")
        println("  " * "-"^50)
    end
    for row in eachrow(case_results)
        println("  $(rpad(row.fair_func, 15)) => shed: $(round(row.pct_shed, digits=2))%, served: $(round(row.pct_served, digits=2))%")
    end
end

csv_path = joinpath(save_dir, "load_shed_comparison.csv")
CSV.write(csv_path, results)
println("\nResults saved to: $csv_path")

# ============================================================
# GENERATE GROUPED BAR CHARTS
# ============================================================
println("\n" * "=" ^ 60)
println("GENERATING BAR CHARTS")
println("=" ^ 60)

for case in CASES
    println("\n  Creating charts for $case...")

    # Active power charts
    create_grouped_bar_chart(case, per_load_results[case], :pshed, save_dir)
    println("    Saved pshed_per_bus_$case.svg")

    create_grouped_bar_chart(case, per_load_results[case], :pd_served, save_dir)
    println("    Saved pd_served_per_bus_$case.svg")
end

# ============================================================
# GENERATE NETWORK PLOTS
# ============================================================
println("\n" * "=" ^ 60)
println("GENERATING NETWORK PLOTS")
println("=" ^ 60)

for case in CASES
    if !haskey(solutions_for_plotting, case)
        continue
    end

    println("\n  Creating network plots for $case...")

    for fair_func in FAIR_FUNCS
        if !haskey(solutions_for_plotting[case], fair_func)
            continue
        end

        sol_data = solutions_for_plotting[case][fair_func]
        mld_solution = sol_data[:mld]
        math_data = sol_data[:mld_math]
        acpf_solution = sol_data[:acpf]
        math_ac = sol_data[:acpf_math]

        # Create network plots only if ACPF converged and voltages are feasible
        if sol_data[:summary]["converged"] && get(sol_data[:summary], "voltage_feasible", true)
            plot_filename = joinpath(save_dir, "network_$(case)_$(fair_func)_mld.svg")
            FairLoadDelivery.plot_network_load_shed(
                mld_solution["solution"],
                math_data;
                output_file=plot_filename
            )
            println("    Saved network_$(case)_$(fair_func)_mld.svg")

            plot_filename = joinpath(save_dir, "network_$(case)_$(fair_func)_acpf.svg")
            FairLoadDelivery.plot_network_load_shed(
                acpf_solution["solution"],
                math_ac;
                output_file=plot_filename, ac_flag=true
            )
            println("    Saved network_$(case)_$(fair_func)_acpf.svg")
        else
            reason = !sol_data[:summary]["converged"] ? "did not converge ($(sol_data[:summary]["termination_status"]))" : "voltage violations detected"
            @warn "[$case/$fair_func] Skipping network plots — $reason"
        end

        # Save solution data as CSV
        solution = mld_solution["solution"]
        ensure_switches_in_solution!(solution, math_data)

        # Load data
        load_df = DataFrame(
            load_id = Int[],
            pd = Float64[],
            qd = Float64[],
            pshed = Float64[],
            qshed = Float64[],
            z_demand = Float64[]
        )
        for (lid, ldata) in solution["load"]
            push!(load_df, (
                parse(Int, lid),
                sum(ldata["pd"]),
                sum(ldata["qd"]),
                sum(ldata["pshed"]),
                sum(ldata["qshed"]),
                ldata["status"]
            ))
        end
        sort!(load_df, :load_id)
        CSV.write(joinpath(save_dir, "solution_load_$(case)_$(fair_func).csv"), load_df)

        # Switch data
        switch_df = DataFrame(
            switch_id = Int[],
            state = Float64[],
            ps_fr = Float64[],
            qs_fr = Float64[]
        )
        for (sid, sdata) in solution["switch"]
            push!(switch_df, (
                parse(Int, sid),
                sdata["state"],
                sum(sdata["pf"]),
                sum(sdata["qf"])
            ))
        end
        sort!(switch_df, :switch_id)
        CSV.write(joinpath(save_dir, "solution_switch_$(case)_$(fair_func).csv"), switch_df)

        # Block data
        block_df = DataFrame(
            block_id = Int[],
            status = Float64[]
        )
        for (bid, bdata) in solution["block"]
            push!(block_df, (
                parse(Int, bid),
                bdata["status"]
            ))
        end
        sort!(block_df, :block_id)
        CSV.write(joinpath(save_dir, "solution_block_$(case)_$(fair_func).csv"), block_df)

        # Bus voltage data
        bus_df = DataFrame(
            bus_id = Int[],
            vm_avg = Float64[],
            z_voltage = Float64[]
        )
        for (bid, bdata) in solution["bus"]
            push!(bus_df, (
                parse(Int, bid),
                mean(sqrt.(bdata["w"])),
                bdata["status"]
            ))
        end
        sort!(bus_df, :bus_id)
        CSV.write(joinpath(save_dir, "solution_bus_$(case)_$(fair_func).csv"), bus_df)

        println("    Saved solution CSVs for $(case)_$(fair_func)")

        # Per-phase voltage CSV
        voltage_by_name = extract_voltage_by_bus_name(acpf_solution["solution"], math_ac)
        volt_phase_df = DataFrame(bus_name = String[], phase = String[], voltage_pu = Float64[])
        phase_map = Dict(1 => "A", 2 => "B", 3 => "C")
        for bus_name in sort(collect(keys(voltage_by_name)))
            for phase in sort(collect(keys(voltage_by_name[bus_name])))
                push!(volt_phase_df, (bus_name, phase_map[phase], voltage_by_name[bus_name][phase]))
            end
        end
        CSV.write(joinpath(save_dir, "solution_bus_voltage_per_phase_$(case)_$(fair_func)_acpf.csv"), volt_phase_df)
        println("    Saved solution_bus_voltage_per_phase_$(case)_$(fair_func)_acpf.csv")
    end
end

# ============================================================
# VOLTAGE PER PHASE COMPARISON PLOTS
# ============================================================
println("\n" * "=" ^ 60)
println("GENERATING VOLTAGE PER PHASE PLOTS")
println("=" ^ 60)

for case in CASES
    println("\n  Creating voltage comparison plot for $case...")

    # Collect voltage data: fair_func => bus_name => phase => voltage
    voltage_data_per_func = Dict{String, Dict{String, Dict{Int, Float64}}}()

    for fair_func in FAIR_FUNCS
        if !haskey(solutions_for_plotting, case) || !haskey(solutions_for_plotting[case], fair_func)
            continue
        end
        sol_data = solutions_for_plotting[case][fair_func]
        voltage_data_per_func[fair_func] = extract_voltage_by_bus_name(
            sol_data[:acpf]["solution"],
            sol_data[:acpf_math]
        )
    end

    if !isempty(voltage_data_per_func)
        plot_voltage_per_bus_comparison(
            voltage_data_per_func,
            joinpath(save_dir, "voltage_per_phase_$case.svg");
            title = "Bus Voltage Per Phase by Fairness Function: $case"
        )
    end
end

# ============================================================
# LOADSHED PER BUS COMPARISON PLOTS
# ============================================================
println("\n" * "=" ^ 60)
println("GENERATING LOADSHED PER BUS PLOTS")
println("=" ^ 60)

for case in CASES
    if !haskey(per_load_results, case)
        continue
    end

    println("\n  Creating loadshed comparison plot for $case...")

    # Build loadshed data from per_load_results: fair_func => (bus_names, pshed_pct, qshed_pct)
    loadshed_data_per_func = Dict{String, Tuple{Vector{String}, Vector{Float64}, Vector{Float64}}}()

    for fair_func in FAIR_FUNCS
        if !haskey(per_load_results[case], fair_func)
            continue
        end
        data = per_load_results[case][fair_func]
        if isempty(data[:load_ids])
            continue
        end
        loadshed_data_per_func[fair_func] = (
            data[:load_names],
            data[:pshed],
            data[:qshed]
        )
    end

    if !isempty(loadshed_data_per_func)
        plot_loadshed_per_bus_comparison(
            loadshed_data_per_func,
            joinpath(save_dir, "loadshed_per_bus_$case.svg");
            title = "Load Shed Per Bus: $case"
        )
    end
end

# ============================================================
# FINAL WEIGHTS COMPARISON
# ============================================================
println("\n" * "=" ^ 60)
println("FINAL WEIGHTS COMPARISON")
println("=" ^ 60)

for case in CASES
    if !haskey(final_weights_results, case)
        continue
    end

    println("\n$case:")

    # Get all load IDs
    all_weight_ids = Set{Int}()
    for (ff, data) in final_weights_results[case]
        union!(all_weight_ids, data[:weight_ids])
    end
    load_ids_sorted = sort(collect(all_weight_ids))

    if isempty(load_ids_sorted)
        println("  No weight data available")
        continue
    end

    # Build weights DataFrame
    weights_df = DataFrame(load_id = load_ids_sorted)
    for fair_func in FAIR_FUNCS
        if !haskey(final_weights_results[case], fair_func) || isempty(final_weights_results[case][fair_func][:weight_ids])
            weights_df[!, fair_func] = fill(NaN, length(load_ids_sorted))
        else
            data = final_weights_results[case][fair_func]
            id_to_weight = Dict(zip(data[:weight_ids], data[:weights]))
            weights_df[!, fair_func] = [haskey(id_to_weight, lid) ? id_to_weight[lid] : NaN for lid in load_ids_sorted]
        end
    end

    # Print weight statistics only
    for fair_func in FAIR_FUNCS
        wts = weights_df[!, fair_func]
        valid_wts = filter(!isnan, wts)
        if !isempty(valid_wts)
            println("  $(rpad(fair_func, 15)): min=$(round(minimum(valid_wts), digits=2)), max=$(round(maximum(valid_wts), digits=2)), spread=$(round(maximum(valid_wts)-minimum(valid_wts), digits=2))")
        end
    end

    # Save CSV
    CSV.write(joinpath(save_dir, "final_weights_$case.csv"), weights_df)

    # Identify critical load IDs from the math dict in solutions_for_plotting
    critical_ids_case = Int[]
    if haskey(solutions_for_plotting, case)
        for (ff, sol) in solutions_for_plotting[case]
            math_case = sol[:mld_math]
            for (lid_str, load) in math_case["load"]
                if get(load, "critical", 0) == 1
                    push!(critical_ids_case, parse(Int, lid_str))
                end
            end
            break  # only need one fair_func to get the load data
        end
    end
    unique!(critical_ids_case)

    # Create bar chart of all weights
    p_weights = plot(xlabel="Load ID", ylabel="Final Weight", title="Final Weights: $case",
                     legend=:outertopright, xticks=load_ids_sorted, size=(1100, 500))
    n_funcs = length(FAIR_FUNCS)
    bar_width = 0.8 / n_funcs
    offsets = collect(range(-(n_funcs-1)/2 * bar_width, (n_funcs-1)/2 * bar_width, length=n_funcs))

    for (i, fair_func) in enumerate(FAIR_FUNCS)
        wts = weights_df[!, fair_func]
        valid_mask = .!isnan.(wts)
        if any(valid_mask)
            bar!(p_weights, load_ids_sorted[valid_mask] .+ offsets[i], wts[valid_mask],
                bar_width=bar_width*0.9, label=FAIR_FUNC_LABELS[fair_func], color=FAIR_FUNC_COLORS[fair_func])
        end
    end
    savefig(p_weights, joinpath(save_dir, "final_weights_$case.svg"))
    println("  Saved: final_weights_$case.csv, final_weights_$case.svg")

    # Create critical load weight plot (side-by-side: all weights + critical-only zoom)
    if !isempty(critical_ids_case)
        crit_mask = [lid in critical_ids_case for lid in load_ids_sorted]
        crit_ids = load_ids_sorted[crit_mask]

        # Left: all weights with critical loads highlighted
        p_all = plot(xlabel="Load ID", ylabel="Final Weight", title="All Weights: $case",
                     legend=:outertopright, xticks=load_ids_sorted, ylims=(0, 10), size=(600, 400))
        for (i, fair_func) in enumerate(FAIR_FUNCS)
            wts = weights_df[!, fair_func]
            valid_mask = .!isnan.(wts)
            if any(valid_mask)
                bar!(p_all, load_ids_sorted[valid_mask] .+ offsets[i], wts[valid_mask],
                    bar_width=bar_width*0.9, label=FAIR_FUNC_LABELS[fair_func], color=FAIR_FUNC_COLORS[fair_func])
            end
        end
        # Mark critical loads on x-axis
        vline!(p_all, crit_ids, color=:red, linestyle=:dash, linewidth=1, label="Critical")

        # Right: zoom on critical loads only
        p_crit = plot(xlabel="Load ID", ylabel="Final Weight", title="Critical Load Weights: $case",
                      legend=:outertopright, xticks=crit_ids, size=(600, 400))
        for (i, fair_func) in enumerate(FAIR_FUNCS)
            wts = weights_df[!, fair_func]
            crit_wts = wts[crit_mask]
            crit_valid = .!isnan.(crit_wts)
            if any(crit_valid)
                bar!(p_crit, crit_ids[crit_valid] .+ offsets[i], crit_wts[crit_valid],
                    bar_width=bar_width*0.9, label=FAIR_FUNC_LABELS[fair_func], color=FAIR_FUNC_COLORS[fair_func])
            end
        end

        p_combined = plot(p_all, p_crit, layout=(1, 2), size=(1400, 500),
                          left_margin=10Plots.mm, bottom_margin=10Plots.mm)
        savefig(p_combined, joinpath(save_dir, "critical_weights_$case.svg"))
        println("  Saved: critical_weights_$case.svg")
    end
end

# ============================================================
# FAIRNESS METRICS SUMMARY CSV
# ============================================================
println("\n" * "=" ^ 60)
println("FAIRNESS METRICS SUMMARY")
println("=" ^ 60)

for case in CASES
    if !haskey(per_load_results, case)
        continue
    end

    println("\n$case:")

    # Get total demand from results DataFrame
    case_rows = filter(row -> row.case == case, results)
    if isempty(case_rows)
        continue
    end
    total_demand = first(case_rows).total_demand

    # Build summary DataFrame
    summary_df = DataFrame(
        Formulation = String[],
        TotalServed_pct = Float64[],
        TotalShed_kW = Float64[],
        MaxBusShed_pct = Float64[],
        MaxBusServed_pct = Float64[],
        MinBusServed_pct = Float64[],
        ShedStdDev_pct = Float64[],
        JainsIndex = Float64[],
        PalmaRatio = Float64[],
        ProportionalFairness = Float64[],
        GiniCoeff = Float64[],
        CV_PctServed = Float64[]
    )

    for fair_func in FAIR_FUNCS
        if !haskey(per_load_results[case], fair_func)
            continue
        end

        data = per_load_results[case][fair_func]
        pshed_vals = data[:pshed]       # per-bus shed PERCENTAGES (0-100)
        pd_vals = data[:pd_served]      # per-bus served PERCENTAGES (0-100)

        if isempty(pshed_vals) || isempty(pd_vals)
            continue
        end

        # Get correct total served % from results DataFrame (kW-based, not from per-bus percentages)
        func_row = filter(row -> row.case == case && row.fair_func == fair_func, results)
        if isempty(func_row)
            continue
        end
        total_served_pct = first(func_row).pct_served
        total_shed_kw = first(func_row).final_pshed

        # Per-bus statistics (on percentages)
        max_shed_pct = maximum(pshed_vals)
        max_served_pct = maximum(pd_vals)
        min_served_pct = minimum(pd_vals)
        shed_stddev = Statistics.std(pshed_vals)

        # Fairness metrics on percent served (normalizes across different bus demand levels)
        if length(pd_vals) >= 2
            jains_idx = FairLoadDelivery.jains_index(pd_vals)
            palma = length(pd_vals) >= 3 ? FairLoadDelivery.palma_ratio(pd_vals) : NaN
            # Proportional fairness: Σ log(served_pct) — only for buses with positive served
            pos_served = filter(x -> x > 0, pd_vals)
            prop_fair = !isempty(pos_served) ? FairLoadDelivery.alpha_fairness(pos_served, 1) : NaN
            gini = FairLoadDelivery.gini_index(pd_vals)
            cv_served = std(pd_vals) / mean(pd_vals)
        else
            jains_idx = NaN
            palma = NaN
            prop_fair = NaN
            gini = NaN
            cv_served = NaN
        end

        push!(summary_df, (
            get(FAIR_FUNC_LABELS, fair_func, fair_func),
            round(total_served_pct, digits=2),
            round(total_shed_kw, digits=2),
            round(max_shed_pct, digits=2),
            round(max_served_pct, digits=2),
            round(min_served_pct, digits=2),
            round(shed_stddev, digits=2),
            round(jains_idx, digits=4),
            round(palma, digits=4),
            round(prop_fair, digits=4),
            round(gini, digits=4),
            round(cv_served, digits=4)
        ))

        println("  $(rpad(get(FAIR_FUNC_LABELS, fair_func, fair_func), 15)): $(total_served_pct)% served, Jain=$(round(jains_idx, digits=4)), Gini=$(round(gini, digits=4)), Palma=$(round(palma, digits=4)), PropFair=$(round(prop_fair, digits=4))")
    end

    # Save summary CSV
    summary_path = joinpath(save_dir, "fairness_summary_$case.csv")
    CSV.write(summary_path, summary_df)
    println("  Saved: $summary_path")
end

if !isempty(failed_combinations)
    println("\n" * "=" ^ 60)
    println("FAILED COMBINATIONS ($(length(failed_combinations)) total)")
    println("=" ^ 60)
    for (case, fair_func, reason) in failed_combinations
        println("  $case / $fair_func: $reason")
    end
end

println("\n" * "=" ^ 60)
println("COMPARISON COMPLETE")
println("Results and charts saved in: $save_dir")
println("=" ^ 60)