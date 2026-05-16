"""
    figure_defaults.jl

    Unified figure font defaults — 9pt, Plots.jl default font family.
    Include from any plotting script via
    `include(joinpath(@__DIR__, "../figure_defaults.jl"))` after `using Plots`
    so the default applies to every subsequent plot call.
"""

using Plots

Plots.default(
    tickfontsize    = 9,
    guidefontsize   = 9,
    titlefontsize   = 9,
    legendfontsize  = 9,
    colorbar_tickfontsize  = 9,
    colorbar_titlefontsize = 9,
    plot_titlefontsize     = 9,
)
