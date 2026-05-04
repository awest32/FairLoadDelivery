#=
Standalone regularization sweep for the lower-level MLD.

Goal: find values of `regularization` (the L2 penalty on pd_var in the lower-level
objective) that make Ipopt converge with LOCALLY_SOLVED at the operating points that
currently fail (LS_PERCENT in {0.7, 0.9}, with and without a critical load).

Does NOT modify FairLoadDelivery source — replicates `build_mc_mld_shedding_implicit_diff`
locally as a closure so we can sweep `regularization` freely.

Run from repo root:
    julia --project=. scratch/reg_sweep.jl
=#

using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution
using PowerModels
using Ipopt
using DiffOpt
using JuMP
import MathOptInterface
const MOI = MathOptInterface

const _PMD = PowerModelsDistribution
const _IM  = PowerModelsDistribution.InfrastructureModels

const FLD = FairLoadDelivery

# ---------- self-contained lower-level build with adjustable regularization ----------
function make_build_fn(reg::Float64)
    return function (pm::_PMD.AbstractUBFModels)
        pm.model = JuMP.Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))

        _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
        FLD.variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

        _PMD.variable_mc_branch_power(pm)
        _PMD.variable_mc_switch_power(pm)
        _PMD.variable_mc_switch_state(pm; relax=true)
        _PMD.variable_mc_shunt_indicator(pm; relax=true)
        _PMD.variable_mc_transformer_power(pm)

        _PMD.variable_mc_gen_indicator(pm; relax=true)
        _PMD.variable_mc_generator_power_on_off(pm)

        _PMD.variable_mc_storage_power_mi_on_off(pm; relax=true, report=true)
        for i in _PMD.ids(pm, :storage)
            z = _PMD.var(pm, :z_storage, i)
            if JuMP.is_binary(z)
                JuMP.unset_binary(z)
                JuMP.set_lower_bound(z, 0.0)
                JuMP.set_upper_bound(z, 1.0)
            end
        end

        _PMD.variable_mc_load_indicator(pm; relax=true)
        FLD.variable_mc_load_shed(pm)
        FLD.variable_block_indicator(pm; relax=true)
        FLD.variable_mc_fair_load_weights(pm)

        _PMD.constraint_mc_model_current(pm)
        for i in _PMD.ids(pm, :ref_buses);  _PMD.constraint_mc_theta_ref(pm, i); end
        _PMD.constraint_mc_bus_voltage_on_off(pm)
        for i in _PMD.ids(pm, :gen);        _PMD.constraint_mc_generator_power(pm, i); end
        for i in _PMD.ids(pm, :bus);        FLD.constraint_mc_power_balance_shed(pm, i); end

        for i in _PMD.ids(pm, :storage)
            _PMD.constraint_storage_state(pm, i)
            _PMD.constraint_storage_complementarity_nl(pm, i)
            _PMD.constraint_mc_storage_losses(pm, i)
            _PMD.constraint_mc_storage_thermal_limit(pm, i)
            FLD.constraint_mc_storage_on_off(pm, i)
        end

        for i in _PMD.ids(pm, :branch)
            _PMD.constraint_mc_power_losses(pm, i)
            FLD.constraint_model_voltage_magnitude_difference_fld(pm, i)
            _PMD.constraint_mc_voltage_angle_difference(pm, i)
            _PMD.constraint_mc_thermal_limit_from(pm, i)
            _PMD.constraint_mc_thermal_limit_to(pm, i)
        end

        for i in _PMD.ids(pm, :switch)
            FLD.constraint_switch_state_on_off(pm, i; relax=true)
            _PMD.constraint_mc_switch_thermal_limit(pm, i)
            FLD.constraint_mc_switch_ampacity(pm, i)
            FLD.constraint_model_switch_voltage_magnitude_difference_fld(pm, i)
        end

        for i in _PMD.ids(pm, :transformer)
            _PMD.constraint_mc_transformer_power(pm, i)
        end

        FLD.constraint_source_voltage_bounds(pm)
        FLD.constraint_mc_isolate_block(pm)
        FLD.constraint_radial_topology(pm)
        FLD.constraint_mc_block_energization_consistency_bigm(pm)
        FLD.constraint_switch_budget(pm)
        FLD.constraint_load_shed_definition(pm)
        FLD.constraint_connect_block_load(pm)
        FLD.constraint_connect_load_bus(pm)
        FLD.constraint_connect_block_gen(pm)
        FLD.constraint_connect_block_voltage(pm)
        FLD.constraint_connect_block_shunt(pm)
        FLD.constraint_connect_block_storage(pm)

        FLD.objective_fairly_weighted_max_load_served_regd(pm; regularization=reg)
    end
end

# ---------- sweep ----------
const CASE_FILE = joinpath(@__DIR__, "..", "data", "pmd_opendss",
                           "case6_unbalanced_switch_good4integer.dss")

function run_one(ls_percent::Float64, reg::Float64;
                 critical_load::Vector{String}=String[])
    switch_rating = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)]) * ls_percent
    eng, math, lbs, critical_id = FLD.setup_network(
        CASE_FILE, ls_percent;
        switch_rating=switch_rating, critical_load=critical_load,
    )

    pm = PowerModelsDistribution.instantiate_mc_model(
        math,
        PowerModelsDistribution.LinDist3FlowPowerModel,
        make_build_fn(reg);
        ref_extensions=[FLD.ref_add_load_blocks!],
    )

    JuMP.set_silent(pm.model)
    JuMP.optimize!(pm.model)

    status   = JuMP.termination_status(pm.model)
    obj      = try JuMP.objective_value(pm.model) catch; NaN end

    # Try forward differentiation — this is the actual test (DiffOpt only works on the
    # well-posed status set, AND requires the IFT preconditions at the optimum).
    diff_ok  = false
    diff_err = ""
    if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)
        try
            wparams = pm.model[:fair_load_weights]
            DiffOpt.empty_input_sensitivities!(pm.model)
            for k in eachindex(wparams)
                DiffOpt.set_forward_parameter(pm.model, wparams[k], 1.0)
            end
            DiffOpt.forward_differentiate!(pm.model)
            diff_ok = true
        catch e
            diff_err = sprint(showerror, e)
        end
    end

    return (status=status, obj=obj, diff_ok=diff_ok, diff_err=diff_err)
end

# Configs to test: failing-without-reg cases vs. baseline-working case.
configs = [
    (ls=0.7, crit=String[],     label="LS=0.7  no critical"),
    (ls=0.9, crit=String[],     label="LS=0.9  no critical"),
    (ls=0.8, crit=["l3"],       label="LS=0.8  L3 critical"),
    (ls=0.8, crit=String[],     label="LS=0.8  no critical (control)"),
]

regs = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

println()
println("=" ^ 100)
println("Regularization sweep — case6_unbalanced_switch_good4integer")
println("Columns: termination | objective | DiffOpt forward OK?")
println("=" ^ 100)

results = Dict{Tuple{Float64,String,Float64}, Any}()
for cfg in configs
    println()
    println("── $(cfg.label) ──")
    @printf "%-10s %-25s %-15s %s\n" "reg" "termination" "objective" "DiffOpt"
    for reg in regs
        try
            r = run_one(cfg.ls, reg; critical_load=cfg.crit)
            results[(cfg.ls, join(cfg.crit, ","), reg)] = r
            tag = r.diff_ok ? "OK" : (isempty(r.diff_err) ? "(skipped)" : "FAIL")
            @printf "%-10.0e %-25s %-15.4f %s\n" reg string(r.status) r.obj tag
        catch e
            @printf "%-10.0e %-25s %-15s %s\n" reg "EXCEPTION" "—" sprint(showerror, e)[1:min(60,end)]
        end
    end
end

println()
println("=" ^ 100)
println("Summary (reg values where DiffOpt forward differentiation succeeded)")
println("=" ^ 100)
for cfg in configs
    successes = [reg for reg in regs
                 if haskey(results, (cfg.ls, join(cfg.crit, ","), reg)) &&
                    results[(cfg.ls, join(cfg.crit, ","), reg)].diff_ok]
    println("$(cfg.label)  →  $(isempty(successes) ? "NONE" : join([@sprintf("%.0e", r) for r in successes], ", "))")
end
println()

using Printf  # for @sprintf above (ensure imported even if Julia hadn't loaded it earlier)
