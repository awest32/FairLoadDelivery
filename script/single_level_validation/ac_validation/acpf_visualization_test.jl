"""
Script to run AC Power Flow (ACPF) for each motivation network case
and generate network visualizations using plot_network_load_shed.

Uses setup_network with high generation capacity so all loads are served,
then runs solve_mc_pf with IVRUPowerModel (standard ACPF, no optimization).
"""

using Revise
using FairLoadDelivery
using MKL
using PowerModelsDistribution, PowerModels
using Ipopt
using HSL_jll
using JuMP
using LinearAlgebra, SparseArrays
using Dates

include("../../../src/implementation/network_setup.jl")
include("../../../src/implementation/visualization.jl")

# ============================================================
# SOLVER
# ============================================================

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

# ============================================================
# CONFIGURATION
# ============================================================

const CASES = ["motivation_a", "motivation_b", "motivation_c"]
const GEN_CAP = 10000.0  # Unconstrained generation — all loads served

# ============================================================
# MAIN LOOP
# ============================================================

for case in CASES
#    case = "motivation_a"
    println("\n" * "="^70)
    println("ACPF VISUALIZATION TEST — $case")
    println("="^70)
    # Setup network
    eng, math, lbs, critical_id = setup_network("ieee_13_aw_edit/$case.dss", GEN_CAP, [])

    # Solve standard AC power flow
    soln = solve_mc_pf(math, IVRUPowerModel, ipopt)
    println("ACPF termination status: $(soln["termination_status"])")

    # Create output folder
    today = Dates.today()
    output_folder = joinpath("results", string(today), case, "ac_validation")
    mkpath(output_folder)

    # Generate network visualization
    output_file = joinpath(output_folder, "acpf_network.svg")
    plot_network_load_shed(soln["solution"], math;
        output_file=output_file,
        layout=:ieee13, ac_flag=true)

    println("Saved: $output_file")
    println("="^70)
end

println("\nAll ACPF visualization tests complete.")
