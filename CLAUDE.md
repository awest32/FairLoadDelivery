# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FairLoadDelivery is a Julia package for fair load shedding optimization in power distribution systems. It implements a bilevel optimization framework that balances efficiency with fairness metrics when determining which loads to shed during capacity shortages.

## Build & Development Commands

```bash
# Activate and install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Start interactive development session
julia --project=.

# In Julia REPL:
using Revise
using FairLoadDelivery
```

## Running Scripts

```julia
# Main FLDP workflow
include("script/FLDP.jl")

# Network setup example
eng, math, lbs, critical_id = setup_network("ieee_13_aw_edit/motivation_b.dss", 0.9, [])

# Solve MLD problems
pm_soln = FairLoadDelivery.solve_mc_mld_switch_integer(math, gurobi)
pm_soln = solve_mc_mld_shed_implicit_diff(math, ipopt)
```

## Architecture

### Bilevel Optimization Framework

- **Upper Level**: Optimizes fairness metrics by adjusting load weights
- **Lower Level**: Solves Minimum Load Delivery (MLD) problem given weights, using implicit differentiation (`DiffOpt`) to compute gradients through the optimization

### Core Components

- `src/core/` - Variable definitions, constraints, and objectives for JuMP models
- `src/prob/` - Problem formulations (OPF, MLD, Power Flow)
- `src/implementation/` - Algorithm implementations:
  - `network_setup.jl` - Parses OpenDSS files, identifies load blocks
  - `lower_level_mld.jl` - Lower-level solver with DiffOpt differentiation
  - `palma_relaxation.jl` - Palma ratio fairness optimization
  - `random_rounding.jl` - Converts relaxed solutions to integer-feasible
  - `other_fair_funcs.jl` - Fairness metrics (Jain, proportional, min-max, efficiency)

### Key Concepts

- **Load Blocks**: Connected regions of loads that can be shed together, enforced by switch topology
- **Three-Phase Unbalanced**: Uses PowerModelsDistribution for realistic distribution system modeling
- **Radiality Constraints**: Ensures tree topology in distribution networks

### Solvers

Preconfigured optimizers are exported from the module:
- `ipopt` - NLP/continuous problems (primary)
- `gurobi` - Mixed-integer programming
- `highs` - Secondary MIP solver
- `juniper` - Mixed-integer nonlinear

### Data Format

Network files use OpenDSS `.dss` format, located in `data/`. Primary test case: `ieee_13_aw_edit/motivation_b.dss`

## Key Patterns

### Reference Extensions

Network preprocessing uses the extension pattern:
```julia
ref_extensions=[ref_add_load_blocks!]
# or for rounded solutions:
ref_extensions=[ref_add_rounded_load_blocks!]
```

### Constants

- `zero_tol = 1e-9` - Numerical tolerance for zero checks
- Module aliases: `_PMD` for PowerModelsDistribution, `_IM` for InfrastructureModels
