# Quick test: Palma only, 2 bilevel iterations, multiperiod.
using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra, SparseArrays
using DiffOpt

const PMD = PowerModelsDistribution

include("../bilevel_validation/validation_utils.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/load_shed_as_parameter.jl")

const LS_PERCENT = 0.8
const SOURCE_PU = 1.03
const N_PERIODS = 3
const LOAD_SCALE_FACTORS = [0.8, 1.0, 0.9]
const ITERATIONS = 2
const critical_buses = []

# Reuse create_multinetwork_data from the comparison script
function create_multinetwork_data(base_math::Dict{String,Any}, n_periods::Int, load_scales::Vector{Float64})
    @assert length(load_scales) == n_periods
    mn_data = Dict{String,Any}(
        "multinetwork" => true,
        "per_unit" => true,
        "data_model" => PMD.MATHEMATICAL,
        "nw" => Dict{String,Any}()
    )
    for key in ["baseMVA", "basekv", "bus_lookup", "settings"]
        if haskey(base_math, key)
            mn_data[key] = deepcopy(base_math[key])
        end
    end
    for t in 1:n_periods
        nw_id = string(t - 1)
        nw_data = deepcopy(base_math)
        delete!(nw_data, "multinetwork")
        scale = load_scales[t]
        for (load_id, load) in nw_data["load"]
            load["pd"] = load["pd"] .* scale
            load["qd"] = load["qd"] .* scale
        end
        nw_data["time_period"] = t
        nw_data["load_scale"] = scale
        mn_data["nw"][nw_id] = nw_data
    end
    return mn_data
end

# Setup
println("=" ^ 60)
println("PALMA-ONLY MULTIPERIOD TEST (2 iterations)")
println("=" ^ 60)

eng, math, lbs, critical_id = FairLoadDelivery.setup_network("ieee_13_aw_edit/motivation_c.dss", LS_PERCENT, SOURCE_PU, critical_buses)
sorted_load_ids = sort(parse.(Int, collect(keys(math["load"]))))
n_loads = length(sorted_load_ids)
fair_weights = Float64[math["load"][string(i)]["weight"] for i in sorted_load_ids]
mn_data = create_multinetwork_data(math, N_PERIODS, LOAD_SCALE_FACTORS)

println("Loads: $n_loads, Periods: $N_PERIODS, Total pshed dim: $(n_loads * N_PERIODS)")
println("Initial weights: $fair_weights")

mn_new = deepcopy(mn_data)

for k in 1:ITERATIONS
    global fair_weights, mn_new
    println("\n" * "=" ^ 40)
    println("BILEVEL ITERATION $k")
    println("=" ^ 40)

    # Lower level
    dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs = lower_level_soln_mn(mn_new, fair_weights, k)

    println("  pshed_val range: [$(minimum(pshed_val)), $(maximum(pshed_val))]")
    println("  Jacobian size: $(size(dpshed))")

    # Build pd from refs (matching lower-level constraint source)
    pd_all = Float64[]
    for (nw, lid) in pshed_nw_ids
        push!(pd_all, sum(refs[nw][:load][lid]["pd"]))
    end
    println("  pd_all range: [$(minimum(pd_all)), $(maximum(pd_all))]")

    # Check alignment
    n_violations = count(pshed_val[j] > pd_all[j] + 1e-6 for j in 1:length(pshed_val))
    n_below = count(pshed_val[j] < 0 for j in 1:length(pshed_val))
    println("  Alignment: $n_violations above pd, $n_below below 0")
    if n_violations > 0
        for j in 1:length(pshed_val)
            if pshed_val[j] > pd_all[j] + 1e-6
                println("    pshed[$j]=$(round(pshed_val[j], sigdigits=6)) > pd[$j]=$(round(pd_all[j], sigdigits=6))")
            end
        end
    end

    # Upper level: Palma
    println("\n  Calling Palma...")
    pshed_new, fair_weight_vals, status = lin_palma_reformulated(dpshed, pshed_val, weight_vals, pd_all, critical_id, weight_ids)

    println("  Palma status: $status")
    if status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.TIME_LIMIT]
        println("  New weights: $(round.(fair_weight_vals, digits=3))")
        println("  Δw max: $(round(maximum(abs.(fair_weight_vals .- fair_weights)), digits=6))")
        fair_weights = fair_weight_vals

        # Update weights in mn_data
        nw_ids = sort(collect(keys(mn_new["nw"])), by=x->parse(Int, x))
        for nw_id in nw_ids
            nw_data = mn_new["nw"][nw_id]
            for (i, w) in zip(weight_ids, fair_weight_vals)
                nw_data["load"][string(i)]["weight"] = w
            end
        end
    else
        println("  FAILED — stopping")
        break
    end
end

println("\n" * "=" ^ 60)
println("TEST COMPLETE")
println("=" ^ 60)
