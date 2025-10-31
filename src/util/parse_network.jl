
using PowerModelsDistribution
using SparseArrays

include("lin_dist_3_flow_util.jl")

# Get into the project parent directory to access the data
cd(joinpath(@__DIR__,".",".."))

# Identify the network file path
#file_path = "data/ieee_13_clean/ieee13.dss"
file_path = "data/case3_balanced_switch.dss"

# Extract the network data from the OpenDSS file using PowerModelsDistribution
network = PowerModelsDistribution.parse_file(file_path; data_model=PowerModelsDistribution.MATHEMATICAL)

# Extract the lengths of the network components
bus_ids = collect(keys(network["bus"]))
branch_ids = collect(keys(network["branch"]))
gen_ids = collect(keys(network["gen"]))
load_ids = collect(keys(network["load"]))
m = length(branch_ids)
n = length(bus_ids)
k = length(gen_ids)
pl = length(load_ids)
num_phases = 3 # Number of phases in the network
M = 10^7 # A large number to represent infinity in the context of power limits
ϵ = 0.5 # A small number to set the voltage deviation limits

# Extract the incidence matrix and line parameters
A_1Θ = zeros(m,n) # Initialize the power transfer matrix
for idx in 1:m # Loop through each branch
    branch = string(idx) # Get the branch ID
    fbus = network["branch"][branch]["f_bus"] # Get the from bus of the branch
    tbus = network["branch"][branch]["t_bus"] # Get the to bus of the branch
    A_1Θ[idx, fbus] = 1 # The from bus is injecting power (1)
    A_1Θ[idx, tbus] = -1 # The to bus is absorbing power (-1)
end

# Convert the incidence matrix to a sparse matrix
#A_1Θ_sparse = sparse(A_1Θ)

# Create the three-phase block incidence matrix
A_3Θ = calc_three_phase_incidence_matrix(network, A_1Θ, num_phases)

# Extract the line parameters for the active and reactive power impedance
(HP_3Θ, HQ_3Θ) = calc_three_phase_impedance_matrix(network, m, n, num_phases)
bdiag_HP_3Θ, bdiag_HQ_3Θ = calc_three_phase_bdiag_impedance_matrix(network, HP_3Θ, HQ_3Θ, m, num_phases)



# Create the vectors for active and reactive power injection limits
pmax = sparse(zeros(num_phases*n)) # Initialize the active power injection limits
qmax = sparse(zeros(num_phases*n)) # Initialize the reactive power injection limits
pmin = sparse(ones(num_phases*n)) # Initialize the minimum active power injection limits
qmin = sparse(ones(num_phases*n)) # Initialize the minimum reactive power injection limits

# Create the vectors of active and reactive power flow limits
Pmax = sparse(zeros(num_phases*m)) # Initialize the active power flow limits
Qmax = sparse(zeros(num_phases*m)) # Initialize the reactive power flow limits
Pmin = sparse(zeros(num_phases*m)) # Initialize the minimum active power flow limits
Qmin = sparse(zeros(num_phases*m)) # Initialize the minimum reactive power flow limits

# Create the vector for squared voltage magnitudes limits
vmax = sparse(zeros(num_phases*n)) # Initialize the squared voltage magnitudes limits
vmin = sparse(zeros(num_phases*n)) # Initialize the minimum squared voltage magnitudes limits


# Generation limits (can be loaded from generator parameters)
pgmax = sparse(zeros(num_phases*n)) # Initialize pgmax
qgmax = sparse(zeros(num_phases*n)) # Initialize qgmax
pgmin = sparse(zeros(num_phases*n)) # Initialize pgmin
qgmin = sparse(zeros(num_phases*n)) # Initialize qgmin
for i in 1:n
    # Check if the generator exists for the bus
    for j in gen_ids
        if network["gen"][j]["gen_bus"] == i
            if network["gen"][j]["pmax"][num_phases] == Inf # Check if the maximum pgeneration is infinite
                pgmax[i:i+2] = M*ones(num_phases) # Set a maximum pgeneration for the bus
                qgmax[i:i+2] = M*ones(num_phases) # Set a maximum qgeneration for the bus
            else
                pgmax[i:i+2] = network["gen"][j]["pmax"] # Set a maximum pgeneration for the bus
                qgmax[i:i+2] = network["gen"][j]["qmax"] # Set a maximum qgeneration for the bus
            end
            # pgmax[i] = network["gen"][j]["pmax"][num_phases] == Inf ? M : network["gen"][j]["pmax"][num_phases] # maximum pgeneration
            # qgmax[i] = network["gen"][j]["qmax"][num_phases] == Inf ? M : network["gen"][j]["qmax"][num_phases] # maximum qgeneration
            # pgmin[i] = network["gen"][j]["pmin"][num_phases] == -Inf ? -M : network["gen"][j]["pmin"][num_phases] # minimum pgeneration
            # qgmin[i] = network["gen"][j]["qmin"][num_phases] == -Inf ? -M : network["gen"][j]["qmin"][num_phases] # minimum qgeneration
        end
    end
end

# Generation costs (can be loaded from generator parameters)
gen_costs_quadratic = sparse(zeros(n)) # Initialize quadratic generation costs
gen_costs_linear = sparse(zeros(n)) # Initialize linear generation costs
gen_costs_constant = sparse(zeros(n)) # Initialize constant generation costs
for i in 1:n
    # Check if the generator exists for the bus
    for j in gen_ids
        if network["gen"][j]["gen_bus"] == i
            gen_costs_quadratic[i] = network["gen"][j]["cost"][1]  # Assuming a single cost value for simplicity
            gen_costs_linear[i] = network["gen"][j]["cost"][2]  # Linear cost
            gen_costs_constant[i] = 0
        end
    end
end

# Voltage limits (can be loaded from bus parameters)
for i in 1:n
	if network["bus"][string.(i)]["bus_type"] == "voltage_source" #|| network["bus"][i]["bus_type"] == "slack"
		vmin[i:i+2,1] = ones(num_phases) # Set a minimum voltage for voltage source buses
		vmax[i:i+2,1] = ones(num_phases) # Set a maximum voltage for voltage source buses
    else
        if network["bus"][string.(i)]["vmax"][1] == Inf # Check if the maximum voltage is infinite
            vmax[i:i+2,1] = M*ones(num_phases) # Set a maximum voltage for other buses
            vmin[i:i+2,1] = -M*ones(num_phases) # Set a minimum voltage for other buses
        else
            vmax[i:i+2,1] = network["bus"][string.(i)]["vmax"] # Set a maximum voltage for other buses
            vmin[i:i+2,1] = network["bus"][string.(i)]["vmin"] # Set a minimum voltage for other buses
        end
    end
end

# vmax = [network["bus"][i]["vmax"][1] == Inf ? M*ones(num_phases) : network["bus"][i]["vmax"] for i in string.(1:n)]
# vmin = [network["bus"][i]["vmin"][1] == -Inf ? M*ones(num_phases) : network["bus"][i]["vmin"] for i in string.(1:n)]


# Load parameters (can be loaded from load parameters)
# Assuming constant power loads for simplicity
pd, qd = extract_three_phase_load_vector(network, n, num_phases) # Extract the active and reactive power loads for each bus



