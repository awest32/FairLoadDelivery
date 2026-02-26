using PowerModelsDistribution

function setup_network(case::String, ls_percent::Float64, critical_load)
    dir = @__DIR__
    casepath = "data/$case"
    file = joinpath(dir, "../../", casepath)

    data = case 
    vscale = 1
    loadscale = 1   

    eng = PowerModelsDistribution.parse_file(file)#, transformations=[PowerModelsDistribution.transform_loops!,PowerModelsDistribution.remove_all_bounds!])

    eng["settings"]["sbase_default"] = 1
    eng["voltage_source"]["source"]["rs"] *=0
    eng["voltage_source"]["source"]["xs"] *=0
    eng["voltage_source"]["source"]["vm"] *=vscale

    "Ensure use the reduce lines function in Fred's basecase script"
    #PowerModelsDistribution.reduce_line_series!(eng)


    math = PowerModelsDistribution.transform_data_model(eng)

    # Store source voltage in per-unit for use in constraints
    math["source_vm_pu"] = eng["voltage_source"]["source"]["vm"]

    # for (idx, switch) in math["switch"]
    #     switch["state"] = 1
    # end
    lbs = PowerModelsDistribution.identify_load_blocks(math)
    # lbs = Dict{String, Any}()
    # lbs_eng = Dict{String, Any}()
    # lb1 = [645, 646, 611, 652, 671]
    # lb2 = [670]#really 632
    # lb3 = [634, 675, 692]
    # lbs_eng[1] = lb1
    # lbs_eng[2] = lb2
    # lbs_eng[3] = lb3

    # for (lb_id, lb) in lbs_eng
    #     lb_vect = []
    #     for bus in lb
    #         for cons in 1:length(ref[:bus_loads][math["bus_lookup"][string(bus)]])
    #             push!(lb_vect, ref[:bus_loads][math["bus_lookup"][string(bus)]][cons])
    #         end
    #     end
    #     lbs[lb_id] = lb_vect
    # end

    get(eng, "time_series", Dict())

    # Update the voltage limits
    for (i,bus) in math["bus"]
            bus["vmax"][:] .= 1.05
            bus["vmin"][:] .= 0.95
    end

    # Update the current limits on the switches based upon the case
    if case == "ieee_13_aw_edit/motivation_a.dss"
        for (i,switch) in math["switch"]
            switch["dispatchable"] = 1.0
            if switch["name"] == "632633"
                switch["current_rating"][:] .= 700#308
            elseif switch["name"] == "632645"
                switch["current_rating"][:] .= 700#322
            end
        end
    elseif case == "ieee_13_aw_edit/motivation_a_with_storage.dss"
        for (i,switch) in math["switch"]
            switch["dispatchable"] = 1.0
            if switch["name"] == "632633"
                switch["current_rating"][:] .= 700#308
            elseif switch["name"] == "632645"
                switch["current_rating"][:] .= 700#322
            end
        end
    elseif case == "ieee_13_aw_edit/motivation_b.dss"
       for (i,switch) in math["switch"]
            switch["dispatchable"] = 1.0
            if switch["name"] == "632633"
                switch["current_rating"][:] .= 700#304
            elseif switch["name"] == "632645"
                switch["current_rating"][:] .= 700#305
            elseif switch["name"] == "671692"
                switch["current_rating"][:] .= 700#80
            end
        end
    elseif case == "ieee_13_aw_edit/motivation_c.dss"
       for (i,switch) in math["switch"]
            switch["dispatchable"] = 1.0
            switch["current_rating"][:] .= 200
            # if switch["name"] == "632633"
            #     switch["current_rating"][:] .= 700#310
            # elseif switch["name"] == "632645"
            #     switch["current_rating"][:] .= 700#264
            # elseif switch["name"] == "671692"
            #     switch["current_rating"][:] .= 700#70   
            # elseif switch["name"] == "646611"
            #     switch["current_rating"][:] .= 700#264
            # elseif switch["name"] == "634675"
            #     switch["current_rating"][:] .= 700
            # elseif switch["name"] == "670671"
            #     switch["current_rating"][:] .= 700
            # end
        end
    elseif case == "ieee_13_aw_edit/motivation_d.dss"
       for (i,switch) in math["switch"]
            switch["dispatchable"] = 1.0
            if switch["name"] == "632633"
                switch["current_rating"][:] .= 700#310
            elseif switch["name"] == "632645"
                switch["current_rating"][:] .= 700#264
            elseif switch["name"] == "671692"
                switch["current_rating"][:] .= 700#70
            elseif switch["name"] == "646611"
                switch["current_rating"][:] .= 700#264
            end
        end
    end
    # Ensure the generation from the source bus is less than the max load
    for (i,gen) in math["gen"]
        if gen["source_id"] == "voltage_source.source"
            pd_phase1=0
            pd_phase2=0
            pd_phase3=0
            qd_phase1=0
            qd_phase2=0
            qd_phase3=0
            for (ind, d) in math["load"]
                # @info d
                # @info d["connections"]
                for (idx, con) in enumerate(d["connections"])
                    # @info "Load at connection $(d["connections"][idx]) has pd=$(d["pd"][idx]) and qd=$(d["qd"][idx])"
                    if 1 == con# d["connections"] 
                        pd_phase1 += d["pd"][idx]
                        qd_phase1 += d["qd"][idx]
                    end
                    if 2 == con
                        pd_phase2 += d["pd"][idx]
                        qd_phase2 += d["qd"][idx]
                    end 
                    if 3 == con
                        pd_phase3 += d["pd"][idx]
                        qd_phase3 += d["qd"][idx]
                    end
                end
            end
            gen["pmax"][1] = pd_phase1 * ls_percent
            gen["qmax"][1] = qd_phase1 * ls_percent
            gen["pmax"][2] = pd_phase2 * ls_percent
            gen["qmax"][2] = qd_phase2 * ls_percent
            gen["pmax"][3] = pd_phase3 * ls_percent
            gen["qmax"][3] = qd_phase3 * ls_percent
            gen["pmin"][:] .= 0
            gen["qmin"][:] .= 0
        end
    end

    # Create the critical load set
    #critical_load = ["675a"]
    critical_id = []
    #critical_load = ["l4"]
    for (i,load) in math["load"]
        if load["name"] in critical_load
            load["critical"] = 1
            load["weight"] = 100000
            push!(critical_id,parse(Int,i))
            #println("Load $(load["name"]) at math load node $(i) is critical.")
        else
            load["critical"] = 0
            load["weight"] = 10
            #println("Load $(load["name"]) at math load node $(i) is not critical.")

        end
    end

    # Set this mapping for the radial topology constraint
    for (switch_id, switch) in enumerate(math["switch"])
        math["switch"][string(switch_id)]["branch_id"] = 0
        for (branch_id, branch) in enumerate(math["branch"])
                if branch[2]["source_id"] == switch[2]["source_id"]
                    switch[2]["branch_id"] = branch_id  # Assuming you have this mapping
                end
        end
    end

    # Declare load blocks in the math model
    bus_block_map = Dict{Int,Int}()
    for (block_id, buses) in enumerate(lbs)
        for bus in buses
            bus_block_map[bus] = block_id
        end
    end
    math["block"] = Dict{String,Any}()
    for (block_id, _) in enumerate(lbs)
        math["block"][string(block_id)] = Dict("id"=>block_id, "state"=>0, "loads"=>Int[])
    end
    for (load_id_str, load) in math["load"]
        block_id = get(bus_block_map, load["load_bus"], 0)
        if block_id > 0
            push!(math["block"][string(block_id)]["loads"], parse(Int, load_id_str))
        end
    end
    return eng, math, lbs, critical_id
end

function update_network(data_in::Dict{String,Any}, block_selection::Dict{}, load_selection::Dict{}, switch_selection::Dict{})
    data = deepcopy(data_in)
    for (switch_id, switch_state) in switch_selection
        data["switch"][string(switch_id)]["dispatchable"] = 1.0
        #@info "Setting switch $switch_id to state $switch_state in math dictionary for round $r"
        data["switch"][string(switch_id)]["state"] = switch_state
        data["switch"][string(switch_id)]["status"] = switch_state
        #@info "Switch $switch_id state in math dictionary is now $(data["switch"][string(switch_id)]["state"])"
    end
    # De-energize load blocks based on the block status from the relaxed solution
    for (load_id, load_data) in data["load"]
        if load_selection[parse(Int,load_id)] <= 0.0
           #@info "De-energizing load $load_id in round $r"
            for phase in 1:length(load_data["pd"])
                 load_data["pd"][phase] = 0.0
                 load_data["qd"][phase] = 0.0
            end
            data["load"][load_id]["vbase"] = 0.0
            data["load"][load_id]["vnom_kv"] = 0.0
            data["load"][load_id]["status"] = 0.0
        end
    end
    # De-energize shunts based on the block status from the relaxed solution
    if !isempty(ref[:shunt])
        for (shunt_id, shunt_data) in data["shunt"]
            block_id = ref[:shunt_block_map][parse(Int,shunt_id)]
            if block_selection[block_id] <= 0.0
                #@info "De-energizing shunt $shunt_id in round $r"
                data["shunt"][shunt_id]["status"] = 0.0
            end
        end
    end
    # De-energize branches based on the block status from the relaxed solution
    for (block_id, branches) in ref[:block_branches]
        for branch_id in branches
           # @info "Branch $branch_id is in block $block_id"
            if block_selection[block_id] <= 0.0
                # @info "De-energizing branch $branch_id in round $r"
                # @info typeof(branch_id)
                data["branch"][string(branch_id)]["status"] = 0.0
                data["branch"][string(branch_id)]["br_status"] = 0.0
                data["branch"][string(branch_id)]["dispatchable"] = 0.0
                data["branch"][string(branch_id)]["vbase"] = 0
            end
        end
    end
    # Ensure the voltages are passed through correctly
    for (bus_id, bus_data) in data["bus"]
        bus_data["vmax"][:] .= 1.05
        bus_data["vmin"][:] .= 0.95
    end

    # Get voltage scale if present (for voltage sensitivity analysis)
    vscale = get(data, "vscale", 1.0)

    for (i,gen) in data["gen"]
        id = parse(Int,i)
        if gen["source_id"] == "voltage_source.source"
            gen["vg"][:] .= ref[:gen][id]["vg"] .* vscale
            gen["vbase"] = ref[:gen][id]["vbase"]
        end
    end

    # Out put if the voltages are close to 1pu, if see voltage drop 10% 5% the will be an issue
    return data
end
#eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/motivation_b.dss", 0.5, ["675a"])
function update_network(data_in::Dict{String,Any}, switch_selection::Dict{}, ref::Dict{Symbol,Any})
    data = deepcopy(data_in)
    for (switch_id, switch_state) in switch_selection
        data["switch"][string(switch_id)]["dispatchable"] = 1.0
        #@info "Setting switch $switch_id to state $switch_state in math dictionary for round $r"
        data["switch"][string(switch_id)]["state"] = switch_state
        data["switch"][string(switch_id)]["status"] = switch_state
        #@info "Switch $switch_id state in math dictionary is now $(data["switch"][string(switch_id)]["state"])"
    end
    # Ensure the voltages are passed through correctly
    for (bus_id, bus_data) in data["bus"]
        bus_data["vmax"][:] .= 1.05
        bus_data["vmin"][:] .= 0.95
    end

    # Get voltage scale if present (for voltage sensitivity analysis)
    vscale = get(data, "vscale", 1.0)

    for (i,gen) in data["gen"]
        id = parse(Int,i)
        if gen["source_id"] == "voltage_source.source"
            gen["vg"][:] .= ref[:gen][id]["vg"] .* vscale
            gen["vbase"] = ref[:gen][id]["vbase"]
        end
    end

    # Out put if the voltages are close to 1pu, if see voltage drop 10% 5% the will be an issue
    return data
end
function update_network(solution_in:: Dict{String,Any}, data_in::Dict{String,Any})
    data = deepcopy(data_in)
    for (switch_id, switch_dict) in solution_in["switch"]
  #      @info "Updating switch $switch_id to state $(switch_dict["state"]) in update_network function"
        data["switch"][string(switch_id)]["dispatchable"] = 1.0
        data["switch"][string(switch_id)]["state"] = switch_dict["state"]
        data["switch"][string(switch_id)]["status"] = switch_dict["state"]
   #     @info "Switch $switch_id state in math dictionary is now $(data["switch"][string(switch_id)]["state"])"
    end

     # Ensure the voltages are passed through correctly
    for (bus_id, bus_data) in data["bus"]
        bus_data["vmax"][:] .= 1.05
        bus_data["vmin"][:] .= 0.95
    end
    
    return data
end

function ac_network_update(data_in::Dict{String,Any}, ref::Dict{Symbol,Any})
    data = deepcopy(data_in)
    # Get voltage scale if present (for voltage sensitivity analysis)
    vscale = get(data, "vscale", 1.0)

    for (i,gen) in data["gen"]
        id = parse(Int,i)
        if gen["source_id"] == "voltage_source.source"
            gen["vg"][:] .= ref[:gen][id]["vg"] .* vscale
            gen["vbase"] = ref[:gen][id]["vbase"]
            # Ensure the generation from the source bus is infinitely large to avoid infeasibility in the AC power flow
            gen["pmax"][:] .= Inf
            gen["qmax"][:] .= Inf
            gen["pmin"][:] .= 0
            gen["qmin"][:] .= 0
            gen["gen_status"] = 1.0
        end
    end
    # Ensure all substation blocks are energized to avoid infeasibility in the AC power flow
    for b in ref[:substation_blocks]
        data["block"][string(b)]["state"] = 1
    end
   

     # Ensure the voltages are passed through correctly

    return data
end