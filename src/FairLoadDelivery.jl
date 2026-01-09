module FairLoadDelivery

import Revise
import MKL
import InfrastructureModels
import PowerModels
import PowerModelsDistribution
import JuMP
import StatsFuns
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
using Dates

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
include("implementation/palma_relaxation.jl")
include("implementation/other_fair_funcs.jl")
include("implementation/random_rounding.jl")
include("implementation/export_results.jl")


export nw_id_default, optimize_model!, ismultinetwork, update_data!, ref_add_load_blocks!, ref_add_rounded_load_blocks!
export solve_mc_opf_acp, solve_mc_pf_aw, solve_mc_mld, solve_mc_mld_switch, solve_mc_mld_shed_implicit_diff, solve_mc_mld_shed_random_round, solve_mc_mld_traditional
export build_mc_opf_ldf, build_mc_pf_switch, build_mc_pf_aw,build_mc_mld_shedding_implicit_diff, build_mc_mld_shedding_random_rounding, build_mc_mld_switchable_relaxed, build_mc_mld_switchable_integer
export ipopt, gurobi, highs, juniper
export setup_network, lower_level_soln,generate_bernoulli_samples, radiality_check, update_network, ac_feasibility_test
export lin_palma, lin_palma_w_grad_input, proportional_fairness_load_shed, complete_efficiency_load_shed, min_max_load_shed, jains_fairness_index
export plot_dpshed_heatmap, plot_load_shed_per_bus, plot_weights_per_load, export_results, create_save_folder

end #module FairLoadDelivery
