"""
    Per-case paper-aligned block display order for the load-shed heatmaps.

    The bilevel + trade-off heatmaps both aggregate per-load shed up to a
    coarser unit for the figure. In case6 that unit happens to be a single
    bus, but conceptually it's a *load block* (the connected region that the
    switch topology toggles on/off together), and the 13-bus case will have
    multi-bus blocks. To keep the figures aligned with the paper's
    numbering — and to support multi-bus blocks down the road — each case
    gets a manual ordering here:

        (paper_block_number, paper_display_label, [bus_names_in_block])

    Rules:
      * `paper_block_number` is the index the paper uses. It need not be
        contiguous on the heatmap; blocks skipped at runtime leave the
        number out (e.g. case6 "primary" has no loads, so column 3 is
        absent and ticks read 1, 2, 4, 5, 6).
      * `bus_names_in_block` is the list of bus names (as they appear in
        `load_bus_names` from the trade-off JLD2 or `bus_name_map` from
        the bilevel pipeline) that belong to this paper block.
      * Empty bus list = block has no loads in any case; it is silently
        dropped from the figure.

    `resolve_block_display(case, load_bus_names)` returns:
      * (display_blocks, load2block) when the case has a mapping —
        `display_blocks` is a `Vector{Tuple{Int,String}}` of the
        (paper_num, label) pairs that survived the skip-empty filter, in
        paper order; `load2block` is `Vector{Int}` of length
        length(load_bus_names) with the column index (into display_blocks)
        for each load (0 if the load's bus is not in any displayed block).
      * `nothing` when no mapping exists for the case — caller falls back
        to its default sort-by-bus-id behavior.

    Keyed by both the trade-off `case` tag (e.g. "more_meshed_6_bus") and
    the bilevel CASE name (e.g. "case6_unbalanced_switch_more_meshed_good4integer")
    since the two pipelines disagree on the case identifier.
"""

const _CASE_BLOCK_DISPLAY = Dict{String, Vector{Tuple{Int,String,Vector{String}}}}(
    "case6_unbalanced_switch_more_meshed_good4integer" => [
        (1, "d",       ["loadbusd"]),
        (2, "a",       ["loadbusa"]),
        (3, "primary", String[]),    # source/feeder block, no loads
        (4, "b",       ["loadbusb"]),
        (5, "c",       ["loadbusc"]),
        (6, "e",       ["loadbuse"]),
    ],
    # Trade-off scripts use a short tag for the same case.
    "more_meshed_6_bus" => [
        (1, "d",       ["loadbusd"]),
        (2, "a",       ["loadbusa"]),
        (3, "primary", String[]),
        (4, "b",       ["loadbusb"]),
        (5, "c",       ["loadbusc"]),
        (6, "e",       ["loadbuse"]),
    ],
    # 13-bus case (motivation_c_good4integer): mapping TBD when the figure
    # is needed. Until added here, the heatmap scripts fall back to their
    # default bus-id-sorted ordering.
)

function _bus_to_paper(entries)
    d = Dict{String,Int}()
    for (paper_num, _, bus_names) in entries
        for bn in bus_names
            d[bn] = paper_num
        end
    end
    d
end

function _build_display_and_mapping(case, bus_names_in_order)
    # Shared core: returns (display_blocks, mapping) where mapping[i] gives
    # the column index (into display_blocks) for the i-th bus name in
    # `bus_names_in_order`. Blocks with no matching bus name are dropped.
    haskey(_CASE_BLOCK_DISPLAY, case) || return nothing
    entries = _CASE_BLOCK_DISPLAY[case]
    bus_to_paper = _bus_to_paper(entries)

    seen = Set{Int}()
    for bn in bus_names_in_order
        haskey(bus_to_paper, bn) && push!(seen, bus_to_paper[bn])
    end

    display_blocks = Tuple{Int,String}[]
    for (paper_num, label, _) in entries
        paper_num in seen && push!(display_blocks, (paper_num, label))
    end

    col_of = Dict(num => k for (k, (num, _)) in enumerate(display_blocks))
    mapping = zeros(Int, length(bus_names_in_order))
    for (i, bn) in enumerate(bus_names_in_order)
        haskey(bus_to_paper, bn) && (mapping[i] = get(col_of, bus_to_paper[bn], 0))
    end

    return (display_blocks, mapping)
end

"""
    resolve_block_display(case, load_bus_names) -> (display_blocks, load2block) | nothing

For per-load aggregation (trade-off heatmap). See module docstring.
"""
function resolve_block_display(case::AbstractString,
                               load_bus_names::AbstractVector{<:AbstractString})
    return _build_display_and_mapping(case, load_bus_names)
end

"""
    resolve_block_display_from_buses(case, bus_labels) -> (display_blocks, bus2block) | nothing

For aggregation that has already been reduced to the bus level (bilevel
loadshed_heatmap_mn.jl, results_block_mn.jl). `bus_labels` is a vector of
unique bus names; `bus2block[i]` is the column index into `display_blocks`
for the i-th bus (0 if the bus isn't in any displayed block).
"""
function resolve_block_display_from_buses(case::AbstractString,
                                          bus_labels::AbstractVector{<:AbstractString})
    return _build_display_and_mapping(case, bus_labels)
end
