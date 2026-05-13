using Revise
using FairLoadDelivery
using PowerModelsDistribution
using Ipopt
using JuMP

# Baseline ACPF probe for motivation_c_good4integer.dss.
# Goal: read per-switch apparent-power flows under full demand (no shedding),
# so we can pick switch current_ratings that make the integer Pareto front
# non-trivial. After running, copy the suggested rating into the
# `motivation_c_good4integer.dss` branch of setup_network in
# src/implementation/network_setup.jl.

case_name = "../../data/ieee_13_aw_edit/motivation_c_good4integer.dss"
dir = @__DIR__
case_path = joinpath(dir, case_name)

eng, math, lbs, critical_id = setup_network(case_path, 1.0)

# setup_network caps gen pmax at ls_percent*pd, but ACPF needs gen = demand + losses.
# Lift the cap so the PF is feasible.
for (_, gen) in math["gen"]
    if gen["source_id"] == "voltage_source.source"
        gen["pmax"][:] .= 1e6
        gen["qmax"][:] .= 1e6
        gen["pmin"][:] .= -1e6
        gen["qmin"][:] .= -1e6
    end
end

# setup_network pins all non-source buses to [0.95, 1.05]. ACPF inside JuMP enforces
# these as hard constraints, which is unrealistic for a meshed baseline. Loosen to
# the .dss vminpu=0.6/vmaxpu=1.4 so the PF actually solves.
for (_, bus) in math["bus"]
    if bus["name"] != "rg60"
        bus["vmax"][:] .= 1.03
        bus["vmin"][:] .= 1.03
    end
end

# Set switches to a known radial configuration (open the redundant alternate paths)
# so the network is a tree. Loops (e.g., 671-692-675-634-633-632-670-671 and
# 632-645-646-611-684-671-670-632) are broken by opening 634675 and 646611.
RADIAL_OPEN = ("634675", "646611")
for (_, sw) in math["switch"]
    sw["state"] = (sw["name"] in RADIAL_OPEN) ? 0 : 1
    sw["status"] = sw["state"]
end

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
pf_soln = PowerModelsDistribution.solve_mc_pf(math, ACRUPowerModel, ipopt)
println("ACPF termination: ", pf_soln["termination_status"])

println("\n=== Switch flows under full-load ACPF ===")
println("(values are P/Q in kW/kVar, |S| in kVA — use |S|_max per switch as the rating)\n")

# Sort by switch name for stable output
sorted_sids = sort(collect(keys(math["switch"])); by = sid -> math["switch"][sid]["name"])

for sid in sorted_sids
    sw = math["switch"][sid]
    if sw["state"] == 1
        sw_soln = pf_soln["solution"]["switch"][sid]
        psw = sw_soln["pf"]
        qsw = sw_soln["qf"]
        name = sw["name"]
        println("Switch $sid ($name):")
        s_phases = Float64[]
        for (idx, phase) in enumerate(sw["f_connections"])
            p = psw[idx]; q = qsw[idx]
            s = sqrt(p^2 + q^2)
            push!(s_phases, s)
            println("  Phase $phase: P=$(round(p, digits=2))  Q=$(round(q, digits=2))  |S|=$(round(s, digits=2))")
        end
        s_max = maximum(s_phases)
        println("  -> suggested current_rating ≈ $(round(s_max * 1.05, digits=1))  (|S|_max × 1.05 margin)")
        println()
    end
end
