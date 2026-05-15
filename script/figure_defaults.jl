"""
    figure_defaults.jl

    Unified figure font defaults — 10pt Arial-equivalent. Include from any
    plotting script via `include(joinpath(@__DIR__, "../figure_defaults.jl"))`
    after `using Plots` so the default applies to every subsequent plot call.
"""

using Plots

Plots.default(
    fontfamily      = "Arial",   # falls back to Helvetica/sans-serif if missing
    tickfontsize    = 10,
    guidefontsize   = 10,
    titlefontsize   = 10,
    legendfontsize  = 10,
    colorbar_tickfontsize  = 10,
    colorbar_titlefontsize = 10,
    plot_titlefontsize     = 10,
)
