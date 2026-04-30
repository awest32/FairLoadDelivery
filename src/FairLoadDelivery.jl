module FairLoadDelivery

import Revise
import MKL
import InfrastructureModels
import PowerModels
import PowerModelsDistribution
import JuMP
import StatsFuns
import Statistics
import Ipopt, Gurobi, HiGHS, Juniper
import HSL_jll
import Memento
import Distributions
import Graphs
import LinearAlgebra, SparseArrays
import Random
import DiffOpt
import Plots
import FrankWolfe
import PowerPlots
import DataFrames
import CSV
import Dates
import GraphPlot
import Colors
import Compose
import Compose: compose, context, line, circle, text, stroke, fill, linewidth, fontsize, draw, SVG, inch, cm, mm, pt, font, hcenter, vcenter, hleft, vtop, vbottom, strokedash
using Cairo

ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = JuMP.optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# To make a bilevel JuMP model, we need to create a BilevelJuMP model here 
juniper = JuMP.optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>ipopt, "mip_solver"=>highs)

const _IM = InfrastructureModels
const _PMD = PowerModelsDistribution

const pmd_it_name = "pmd"
const pmd_it_sym = Symbol(pmd_it_name)

#const M = 1e20
const zero_tol = 1e-9
const TRUST_RADIUS = 0.5

# Explicit imports for later export
import InfrastructureModels: optimize_model!, @im_fields, nw_id_default, ismultinetwork, update_data!
import PowerModelsDistribution.PolyhedralRelaxations: FormulationInfo, _build_univariate_lp_relaxation!

include("core/common.jl")
include("core/objective.jl")
include("core/variable.jl")
include("core/constraint.jl")


include("prob/pf.jl")
include("prob/opf.jl")
include("prob/mld.jl")

include("implementation/network_setup.jl")
include("implementation/lower_level_mld.jl")
include("implementation/load_shed_as_parameter.jl")
include("implementation/other_fair_funcs.jl")
include("implementation/random_rounding.jl")
include("implementation/export_results.jl")
include("implementation/visualization.jl")


export nw_id_default, optimize_model!, ismultinetwork, update_data!, ref_add_load_blocks!, ref_add_rounded_load_blocks!
export solve_mc_opf_acp, solve_mc_pf_aw, solve_mc_mld, solve_mc_mld_switch, solve_mc_mld_shed_implicit_diff, solve_mc_mld_shed_random_round, solve_mc_mld_traditional
export solve_mc_mld_equality_min, solve_mc_mld_equality_min_integer, solve_mc_mld_switch_relaxed, solve_mc_mld_switch_integer
export solve_mc_mld_proportional_fairness, solve_mc_mld_proportional_fairness_integer
export solve_mc_mld_jain, solve_mc_mld_jain_integer, solve_mc_mld_palma, solve_mc_mld_palma_integer
export solve_mn_mc_mld_switch_relaxed, build_mn_mc_mld_switch_relaxed
export solve_mn_mc_mld_shed_implicit_diff, build_mn_mc_mld_shedding_implicit_diff
export solve_mn_mc_mld_switch_integer, build_mn_mc_mld_switch_integer
export solve_mn_mc_mld_min_max_integer, build_mn_mc_mld_min_max_integer
export solve_mn_mc_mld_proportional_fairness_integer, build_mn_mc_mld_proportional_fairness_integer
export solve_mn_mc_mld_jain_integer, build_mn_mc_mld_jain_integer
export solve_mn_mc_mld_palma_integer, build_mn_mc_mld_palma_integer
export objective_mn_fairly_weighted_min_load_shed, objective_mn_min_max
export objective_mn_proportional_fairness_mld, objective_mn_jain_mld, objective_mn_palma_mld
export weighted_shed_mn, jain_mn, min_max_mn, proportional_mn, palma_mn
export diff_forward_full_jacobian_mn, lower_level_soln_mn
export build_mc_opf_ldf, build_mc_pf_switch, build_mc_pf_aw,build_mc_mld_shedding_implicit_diff, build_mc_mld_shedding_random_rounding, build_mc_mld_switchable_relaxed, build_mc_mld_switchable_integer
export build_mc_mld_proportional_fairness, build_mc_mld_proportional_fairness_integer, objective_proportional_fairness_mld
export build_mc_mld_jain, build_mc_mld_jain_integer, objective_jain_mld, objective_jain_mld_minvar
export build_mc_mld_palma, build_mc_mld_palma_integer, objective_palma_mld, objective_palma_mld_maxmin
export build_mc_mld_min_max, build_mc_mn_mld_min_max, build_mc_mld_min_max_integer, build_mc_mn_mld_min_max_integer
export TRUST_RADIUS
export setup_network, lower_level_soln,generate_bernoulli_samples, radiality_check, update_network, ac_feasibility_test, ac_network_update, ensure_switches_in_solution!
export extract_switch_block_states, find_best_mld_solution, set_source_gen_capacity!, round_and_select_topology_mn
export lin_palma_reformulated, lin_palma_w_grad_input, proportional_fairness_load_shed, efficient_load_shed, infinity_norm_fairness_load_shed, jains_fairness_index, min_max_load_shed
export plot_dpshed_heatmap, plot_load_shed_per_bus, plot_weights_per_load, export_results, create_save_folder, visualize_network_svg, plot_network_load_shed, build_load_name_map, build_bus_name_maps
export get_bus_voltage_per_phase, extract_voltage_by_bus_name, extract_per_bus_loadshed
export plot_voltage_per_bus_comparison, plot_loadshed_per_bus_comparison, create_grouped_bar_chart
export plot_fairness_efficiency_pareto, plot_loadshed_distribution_comparison
export FAIR_FUNC_COLORS, FAIR_FUNC_LABELS, FAIR_FUNC_MARKERS

end #module FairLoadDelivery
