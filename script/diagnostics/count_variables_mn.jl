"""
    Count binary vs continuous variables in the multi-period FALD MIPs
    (min-max and Palma ratio) for the 6-bus case6 with T=24.

    Builds the JuMP models without solving and queries variable counts.

    Usage:
        julia --project=. script/diagnostics/count_variables_mn.jl
"""

using FairLoadDelivery
using PowerModelsDistribution
using Gurobi
using Ipopt
using JuMP
using Printf
import MathOptInterface as MOI

const PMD = PowerModelsDistribution

CASE_FILE = joinpath(@__DIR__, "..", "..", "data", "pmd_opendss",
                     "case6_unbalanced_switch_more_meshed_good4integer.dss")
LS_PERCENT = 0.8
N_PERIODS  = 24
SELECTED_HOURS = collect(0:23)
PEAK_TIME_COSTS = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                   for h in SELECTED_HOURS]

println("== Setting up network ==")
eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT;
    switch_rating = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)]) * LS_PERCENT)

mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = SELECTED_HOURS, peak_stress = 1.0, center_at_nominal = true)

n_loads = length(mn_data["nw"][first(keys(mn_data["nw"]))]["load"])
println("  T = $N_PERIODS, N (loads/period) = $n_loads, T·N = $(N_PERIODS * n_loads)")

"""
Build a model with the given build_fn, then count variables by type.
Skip solving — only the JuMP structure is needed.
"""
function count_vars(build_fn, mn_data; label::String)
    pm = instantiate_mc_model(mn_data, PMD.LinDist3FlowPowerModel, build_fn;
        multinetwork = true,
        ref_extensions = [FairLoadDelivery.ref_add_load_blocks!])
    model = pm.model

    total    = num_variables(model)
    n_bin    = num_constraints(model, VariableRef, MOI.ZeroOne)
    n_int    = num_constraints(model, VariableRef, MOI.Integer)
    cont     = total - n_bin - n_int

    n_constr = sum(num_constraints(model, F, S)
                   for (F, S) in list_of_constraint_types(model))

    println("\n== $label ==")
    println("  total variables:      $total")
    println("    binary:             $n_bin")
    println("    general integer:    $n_int")
    println("    continuous:         $cont")
    println("  total constraints:    $n_constr")
    return (total = total, binary = n_bin, integer = n_int,
            continuous = cont, constraints = n_constr)
end

build_min_max = pm -> FairLoadDelivery.build_mn_mc_mld_min_max_integer(pm;
    peak_time_costs = PEAK_TIME_COSTS, alpha = 1.0)
build_palma   = pm -> FairLoadDelivery.build_mn_mc_mld_palma_integer(pm;
    peak_time_costs = PEAK_TIME_COSTS, alpha = 1.0)

mm_sl = count_vars(build_min_max, mn_data; label = "single-level min-max (integer) — case6 T=24")
pa_sl = count_vars(build_palma,   mn_data; label = "single-level Palma  (integer) — case6 T=24")

# ============================================================
# BILEVEL SUB-PROBLEMS
# ============================================================
"""
Build a JuMP model the same way an upper-level subproblem does, then count.
We don't solve — just inspect the structure. Dummy inputs of the right shape.
"""
function count_upper_min_max(m::Int, n_periods::Int, n::Int)
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, weights_new[1:m])
    for id in 1:m
        @constraint(model, weights_new[id] >= 1.0)
        @constraint(model, weights_new[id] <= 10.0)
    end
    @constraint(model, [i=1:m], weights_new[i] <= 0.5)   # placeholder for trust radius
    @constraint(model, [i=1:m], weights_new[i] >= -0.5)
    @variable(model, max_shed >= 0)
    for t in 1:n_periods, i in 1:n
        @constraint(model, max_shed >= 0.0)
    end
    @objective(model, Min, max_shed)
    return _summary(model)
end

"""
Mirror of palma_ratio_minimization_formal_cc structure (formal CC MILP).
Counts what the upper level adds *on top of* the lower-level Jacobian inputs.
"""
function count_upper_palma_formal_cc(m::Int, n_periods::Int, n::Int)
    model = JuMP.Model(Gurobi.Optimizer)
    @variable(model, σ[1:n_periods] >= 1e-8)
    a = [@variable(model, [1:n, 1:n], Bin, base_name = "a_$t") for t in 1:n_periods]
    @variable(model, Δw_z[1:m])
    u_z = [@variable(model, [1:n, 1:n], lower_bound = 0, base_name = "u_z_$t") for t in 1:n_periods]
    # Doubly-stochastic + indicator + sort + denominator constraints (count only).
    for t in 1:n_periods
        for i in 1:n; @constraint(model, sum(a[t][i, j] for j in 1:n) == 1); end
        for j in 1:n; @constraint(model, sum(a[t][i, j] for i in 1:n) == 1); end
        for i in 1:n, j in 1:n
            @constraint(model, a[t][i, j] => {u_z[t][i, j] == 0.0})
            @constraint(model, !a[t][i, j] => {u_z[t][i, j] == 0.0})
        end
        for k in 1:(n - 1)
            @constraint(model,
                sum(u_z[t][k, j] for j in 1:n) <= sum(u_z[t][k+1, j] for j in 1:n))
        end
        @constraint(model, sum(u_z[t][i, j] for i in 1:floor(Int, 0.4n), j in 1:n) == 1)
    end
    for j in 1:m
        t = ((j - 1) ÷ n) + 1
        @constraint(model, Δw_z[j] >= -0.5 * σ[t])
        @constraint(model, Δw_z[j] <=  0.5 * σ[t])
    end
    @objective(model, Min, σ[1])
    return _summary(model)
end

function _summary(model)
    total = num_variables(model)
    n_bin = num_constraints(model, VariableRef, MOI.ZeroOne)
    n_int = num_constraints(model, VariableRef, MOI.Integer)
    cont  = total - n_bin - n_int
    n_constr = sum(num_constraints(model, F, S)
                   for (F, S) in list_of_constraint_types(model))
    return (total = total, binary = n_bin, integer = n_int,
            continuous = cont, constraints = n_constr)
end

println()
println("== Bi-level: lower-level relaxed MLD (DiffOpt, per iter × T·N forward solves) ==")
# build_mn_mc_mld_shedding_implicit_diff: relaxed switch / block / load shed.
lower_ll = count_vars(
    pm -> FairLoadDelivery.build_mn_mc_mld_shedding_implicit_diff(pm),
    mn_data; label = "lower-level relaxed MLD (continuous)")

ul_mm = count_upper_min_max(216, N_PERIODS, n_loads)
println("\n== Bi-level: upper-level min-max (LP, per iter) ==")
@printf "  total=%d binary=%d cont=%d constraints=%d\n" ul_mm.total ul_mm.binary ul_mm.continuous ul_mm.constraints

ul_pa = count_upper_palma_formal_cc(216, N_PERIODS, n_loads)
println("\n== Bi-level: upper-level Palma formal-CC (MILP, per iter) ==")
@printf "  total=%d binary=%d cont=%d constraints=%d\n" ul_pa.total ul_pa.binary ul_pa.continuous ul_pa.constraints

println("\n========================= SUMMARY =========================")
@printf "  %-50s %10s %10s %10s %12s\n" "subproblem" "total" "binary" "cont." "constraints"
@printf "  %-50s %10d %10d %10d %12d\n" "single-level min-max (integer)"     mm_sl.total mm_sl.binary mm_sl.continuous mm_sl.constraints
@printf "  %-50s %10d %10d %10d %12d\n" "single-level Palma   (integer)"     pa_sl.total pa_sl.binary pa_sl.continuous pa_sl.constraints
@printf "  %-50s %10d %10d %10d %12d\n" "bi-level lower-level (relaxed MLD)" lower_ll.total lower_ll.binary lower_ll.continuous lower_ll.constraints
@printf "  %-50s %10d %10d %10d %12d\n" "bi-level upper-level min-max (LP)"  ul_mm.total ul_mm.binary ul_mm.continuous ul_mm.constraints
@printf "  %-50s %10d %10d %10d %12d\n" "bi-level upper-level Palma (MILP)"  ul_pa.total ul_pa.binary ul_pa.continuous ul_pa.constraints
