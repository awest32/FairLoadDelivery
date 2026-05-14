#=
Bilevel Palma trade-off — 3-point alpha sweep
=============================================

Mirrors `run_validation.jl` but is purpose-built for the Palma upper level
with an outer alpha sweep at 3 points (α = 0, 0.5, 1). For each alpha we
run the standard FLDP bilevel loop:

    integer MLD → updated topology → relaxed lower-level (DiffOpt Jacobian)
    → upper-level Palma+α weight update → repeat

then record the final pshed-per-load and the Palma ratio. The functionality
of every step already lives in run_validation.jl (FAIR_FUNC == "palma");
this script is a focused trade-off driver.

Notes:
  - Only the absolute pshed variant is meaningful for integer load-shed
    decisions (proportional collapses on binary fractions).
  - Charnes-Cooper σ·bot=1 is always active inside `lin_palma_reformulated`,
    so α = 0 is the same edge case noted in `palma_trade_off.jl`.
=#

using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS, Juniper
using HSL_jll
using Plots
using Random
using Distributions
using DiffOpt
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra, SparseArrays
using DataFrames
using CSV
using Dates
using Logging, LoggingExtras

include("validation_utils.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/load_shed_as_parameter.jl")

# ============================================================
# Configuration
# ============================================================
const CASE = "case6_unbalanced_switch_meshed_good4integer"
const CASE_FILE = joinpath(@__DIR__, "../../data/pmd_opendss/$CASE.dss")

LS_PERCENT = 0.8
const ITERATIONS = 2
const ALPHAS = [0.0, 0.5, 1.0]
switch_rating = sqrt.([(26.0^2 + 13.1^2), (23.0^2 + 9^2), (21.0^2 + 9.5^2)]) * LS_PERCENT

ipopt_solver  = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi_solver = Gurobi.Optimizer

save_dir = "results/$(Dates.today())/bilevel_palma/$CASE"
mkpath(save_dir)

log_file = joinpath(save_dir, "bilevel_palma.log")
global_logger(TeeLogger(global_logger(), FileLogger(log_file)))
@info "Logging to $log_file"

# ============================================================
# Palma helpers (local; same definition as palma_trade_off.jl)
# ============================================================
function compute_palma_indices(n::Int)
    n_bot = max(1, floor(Int, 0.4 * n))
    n_top = max(1, ceil(Int, 0.1 * n))
    return collect((n - n_top + 1):n), collect(1:n_bot)
end

function palma_ratio_value(pshed::AbstractVector; eps_denom::Float64 = 1e-6)
    n = length(pshed)
    s = sort(collect(pshed))
    top, bot = compute_palma_indices(n)
    num = sum(max(0.0, s[i]) for i in top)
    den = sum(max(0.0, s[i]) for i in bot)
    return den < eps_denom ? Inf : num / den
end

# ============================================================
# Network setup
# ============================================================
print_validation_header("Network Setup")
eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT;
    switch_rating = switch_rating)

n_loads = length(math["load"])
@info "Loaded $CASE — $n_loads loads, $(length(math["switch"])) switches"

# ============================================================
# Bilevel loop, swept across alpha
# ============================================================
results_by_alpha = Dict{Float64, Any}()

for alpha in ALPHAS
    print_validation_header("Alpha = $alpha")

    math_new = deepcopy(math)

    # Initial weights from network
    fair_weights = Float64[load["weight"] for (_, load) in math_new["load"]]

    pshed_lower_traj    = Float64[]
    pshed_upper_traj    = Float64[]
    palma_lower_traj    = Float64[]
    palma_upper_traj    = Float64[]
    final_pshed_per_load = Float64[]
    final_pshed_ids      = Int[]
    final_weight_ids     = Int[]
    statuses             = String[]

    for k in 1:ITERATIONS
        @info "  --- alpha=$alpha  iteration $k ---"

        # Solve integer MLD to fix topology, then update math
        mld_int = solve_mc_mld_switch_integer(math_new, Gurobi.Optimizer)
        math_updated = update_network(mld_int["solution"], math_new)

        # Lower level (DiffOpt Jacobian)
        dpshed_k, pshed_val_k, pshed_ids_k, weight_vals_k, weight_ids_k, _ =
            lower_level_soln(math_updated, fair_weights, k)

        # Upper level: Palma with alpha
        pd_k = Float64[sum(math_new["load"][string(i)]["pd"]) for i in pshed_ids_k]
        pshed_new, fair_weight_vals, status = lin_palma_reformulated(
            dpshed_k, pshed_val_k, weight_vals_k, pd_k;
            critical_ids = critical_id,
            weight_ids   = weight_ids_k,
            alpha        = alpha,
        )

        @info "    upper-level status: $status"
        push!(statuses, string(status))

        # Update weights in math dict
        for (i, w) in zip(weight_ids_k, fair_weight_vals)
            math_new["load"][string(i)]["weight"] = w
        end

        push!(pshed_lower_traj, sum(pshed_val_k))
        push!(pshed_upper_traj, sum(pshed_new))
        push!(palma_lower_traj, palma_ratio_value(pshed_val_k))
        push!(palma_upper_traj, palma_ratio_value(pshed_new))

        @info "    lower total = $(round(sum(pshed_val_k), digits=3)) (Palma=$(round(palma_lower_traj[end], digits=3)))   upper total = $(round(sum(pshed_new), digits=3)) (Palma=$(round(palma_upper_traj[end], digits=3)))"

        final_pshed_per_load = pshed_val_k
        final_pshed_ids      = pshed_ids_k
        final_weight_ids     = weight_ids_k
        fair_weights         = fair_weight_vals
    end

    results_by_alpha[alpha] = Dict(
        "pshed_lower_traj"    => pshed_lower_traj,
        "pshed_upper_traj"    => pshed_upper_traj,
        "palma_lower_traj"    => palma_lower_traj,
        "palma_upper_traj"    => palma_upper_traj,
        "statuses"            => statuses,
        "final_pshed_per_load" => final_pshed_per_load,
        "final_pshed_ids"     => final_pshed_ids,
        "final_weight_ids"    => final_weight_ids,
        "final_total_shed"    => sum(final_pshed_per_load),
        "final_max_shed"      => maximum(final_pshed_per_load),
        "final_palma"         => palma_ratio_value(final_pshed_per_load),
    )
end

# ============================================================
# Save sweep summary CSV
# ============================================================
sweep_df = DataFrame(
    alpha             = Float64[],
    final_total_shed  = Float64[],
    final_max_shed    = Float64[],
    final_palma       = Float64[],
)
for alpha in ALPHAS
    r = results_by_alpha[alpha]
    push!(sweep_df, (alpha, r["final_total_shed"], r["final_max_shed"], r["final_palma"]))
end
CSV.write(joinpath(save_dir, "bilevel_palma_sweep.csv"), sweep_df)

# Per-load CSV (rows = alphas, cols = loads)
load_ids_ref = results_by_alpha[ALPHAS[1]]["final_pshed_ids"]
perload_df = DataFrame(alpha = Float64[])
for lid in load_ids_ref
    perload_df[!, Symbol("load_$(lid)")] = Float64[]
end
for alpha in ALPHAS
    row = vcat(alpha, results_by_alpha[alpha]["final_pshed_per_load"])
    push!(perload_df, row)
end
CSV.write(joinpath(save_dir, "bilevel_palma_perload.csv"), perload_df)

# ============================================================
# Plots: per-alpha trajectory + Pareto + Palma vs alpha + per-load bars
# ============================================================
for alpha in ALPHAS
    r = results_by_alpha[alpha]
    iters = collect(1:length(r["pshed_lower_traj"]))

    p1 = plot(iters, r["pshed_lower_traj"], label = "lower (sum pshed)", lw = 2, marker = :circle,
              xlabel = "iteration", ylabel = "total shed (kW)",
              title  = "alpha = $alpha")
    plot!(p1, iters, r["pshed_upper_traj"], label = "upper (predicted)", lw = 2, marker = :square)

    p2 = plot(iters, r["palma_lower_traj"], label = "lower Palma", lw = 2, marker = :circle,
              xlabel = "iteration", ylabel = "Palma ratio")
    plot!(p2, iters, r["palma_upper_traj"], label = "upper Palma", lw = 2, marker = :square)

    savefig(plot(p1, p2, layout = (1, 2), size = (1200, 400)),
        joinpath(save_dir, "trajectory_alpha_$(alpha).svg"))
end

total_sheds = [results_by_alpha[a]["final_total_shed"] for a in ALPHAS]
max_sheds   = [results_by_alpha[a]["final_max_shed"]   for a in ALPHAS]
palmas      = [results_by_alpha[a]["final_palma"]      for a in ALPHAS]

p_pareto = plot(total_sheds, max_sheds, lw = 2, marker = :circle,
    marker_z = collect(ALPHAS), color = :cividis, colorbar_title = "alpha",
    xlabel = "final total shed (kW)", ylabel = "final max shed (kW)",
    title  = "Bilevel Palma Pareto (3-point sweep)", legend = false)

# Annotate alpha next to each point
for (i, a) in enumerate(ALPHAS)
    annotate!(p_pareto, total_sheds[i], max_sheds[i],
        text("  α=$(a)", 8, :left))
end

p_palma = plot(collect(ALPHAS), palmas, lw = 2, marker = :diamond,
    xlabel = "alpha", ylabel = "Palma ratio (top10% / bot40%)",
    title  = "Final Palma vs alpha", legend = false)

savefig(plot(p_pareto, p_palma, layout = (1, 2), size = (1200, 400)),
    joinpath(save_dir, "pareto_summary.svg"))

# Per-load distribution at each alpha (one combined plot)
load_names = [math["load"][string(lid)]["name"] for lid in load_ids_ref]
dist_plots = Plots.Plot[]
for alpha in ALPHAS
    pshed_per_load = results_by_alpha[alpha]["final_pshed_per_load"]
    p = bar(load_names, pshed_per_load,
        xlabel = "load", ylabel = "shed (kW)",
        title  = "alpha = $alpha",
        legend = false, color = :steelblue, linecolor = :black, xrotation = 45)
    push!(dist_plots, p)
end
savefig(plot(dist_plots..., layout = (1, length(ALPHAS)), size = (400 * length(ALPHAS), 400)),
    joinpath(save_dir, "loadshed_distributions.svg"))

# ============================================================
# Console summary
# ============================================================
println("\n========== Bilevel Palma sweep summary ==========")
println(sweep_df)
println("\nResults written to: ", save_dir)
