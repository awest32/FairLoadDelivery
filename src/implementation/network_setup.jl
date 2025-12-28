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

    for (i,bus) in math["bus"]

            bus["vmax"][:] .= 1.1
            bus["vmin"][:] .= 0.9
    end

    # Ensure the generation from the source bus is less than the max load
    # First calculate the total load
    served = [] #Dict{Any,Any}()
    #ls_percent = 0.5
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
            load["critical"] = 0
            load["weight"] = 10
            push!(critical_id,parse(Int,i))
            #println("Load $(load["name"]) at math load node $(i) is critical.")
        else
            load["critical"] = 0
            load["weight"] = 10
            #println("Load $(load["name"]) at math load node $(i) is not critical.")

        end
    end

    for (switch_id, switch) in enumerate(math["switch"])
        math["switch"][string(switch_id)]["branch_id"] = 0
        for (branch_id, branch) in enumerate(math["branch"])
                if branch[2]["source_id"] == switch[2]["source_id"]
                    switch[2]["branch_id"] = branch_id  # Assuming you have this mapping
                end
        end
    end

    # Declare load blocks in the math model
    math["block"] = Dict{String,Any}()
    for (block, loads) in enumerate(lbs)
        math["block"][string(block)] = Dict("id"=>block, "state"=>0)
    end
    return eng, math, lbs, critical_id
end

#eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/motivation_b.dss", 0.5, ["675a"])