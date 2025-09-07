import pypsa
import os
import pandas as pd

def fix_artificial_lines_reasonable(network):
    """
    Fix artificial lines with reasonable capacity values:
    - s_nom = based on connected bus demand (with safety factor)
    - s_nom_extendable = False (non-extendable)
    - Keep capacity high enough to meet demand
    """
    print("=== FIXING ARTIFICIAL LINES WITH REASONABLE CAPACITY ===")

    # Find artificial lines
    artificial_lines = [line for line in network.lines.index
                       if any(keyword in str(line).lower() for keyword in ['new', '<->', 'artificial'])]

    if not artificial_lines:
        # If no artificial lines found by name, look for lines with s_nom=0
        # which is often a sign of artificial lines
        zero_capacity_lines = network.lines[network.lines.s_nom == 0].index.tolist()
        if zero_capacity_lines:
            artificial_lines = zero_capacity_lines

    print(f"Found {len(artificial_lines)} artificial lines to fix:")

    # Get maximum demand per bus across all snapshots
    bus_max_demand = {}
    for bus in network.buses.index:
        bus_demand = 0
        for load_name, load in network.loads.iterrows():
            if load.bus == bus and load_name in network.loads_t.p_set.columns:
                bus_demand = max(bus_demand, network.loads_t.p_set[load_name].max())
        bus_max_demand[bus] = bus_demand

    # Fix each artificial line with reasonable capacity
    for line_name in artificial_lines:
        # Get connected buses
        bus0 = network.lines.loc[line_name, 'bus0']
        bus1 = network.lines.loc[line_name, 'bus1']

        # Get maximum demand at these buses
        bus0_demand = bus_max_demand.get(bus0, 0)
        bus1_demand = bus_max_demand.get(bus1, 0)

        # Calculate required capacity with safety factor
        # Use 3x the higher demand to ensure adequate capacity
        safety_factor = 3.0
        required_capacity = max(bus0_demand, bus1_demand) * safety_factor

        # Ensure minimum reasonable capacity (1000 MW)
        required_capacity = max(required_capacity, 1000)

        print(f"\n Fixing: {line_name}")
        print(f"    Connected buses: {bus0} ↔ {bus1}")
        print(f"    Bus demands: {bus0}: {bus0_demand:.1f} MW, {bus1}: {bus1_demand:.1f} MW")

        # Set s_nom to required capacity
        old_s_nom = network.lines.loc[line_name, 's_nom']
        network.lines.loc[line_name, 's_nom'] = required_capacity
        print(f"    s_nom: {old_s_nom} → {required_capacity:.1f} MW")

        # Make sure line is not extendable
        if 's_nom_extendable' not in network.lines.columns:
            network.lines['s_nom_extendable'] = False
        network.lines.loc[line_name, 's_nom_extendable'] = False
        print(f"    s_nom_extendable: → False")

    return network

def remove_offshore_wind(network):
    """
    Remove offshore wind generators. 
    All of these have zero nominal capacity (likely missing data). 
    Need to remove them to avoid division by zero error in constraint check for slack gens.
    Problem is still feasible without offwind slack since pypsa optimize still feasible.
    """
    
    # First, identify offshore wind generators
    offwind_gens = network.generators[
        network.generators.index.str.contains('offwind', case=False, na=False)
    ].index
    
    print(f"Found {len(offwind_gens)} offshore wind generators:")
    print(offwind_gens.tolist())
    
    # Check their properties
    offwind_data = network.generators.loc[offwind_gens, ['p_nom', 'control', 'carrier']]
    print("\nOffshore wind generator details:")
    print(offwind_data)
    
    # Remove offshore wind generators one by one
    print(f"\nRemoving {len(offwind_gens)} offshore wind generators...")
    for gen in offwind_gens:
        network.remove("Generator", gen)

def create_pypsa_network(network_file):
    """Create a PyPSA network from the .nc file."""
    # Initialize network
    network = pypsa.Network(network_file)
    for storage_name in network.storage_units.index:
        # Use .loc for direct assignment to avoid SettingWithCopyWarning
        network.storage_units.loc[storage_name, 'cyclic_state_of_charge'] = False

        # Set marginal_cost to 0.01
        network.storage_units.loc[storage_name, 'marginal_cost'] = 0.01

        # Set marginal_cost_storage to 0.01
        network.storage_units.loc[storage_name, 'marginal_cost_storage'] = 0.01

        # Set spill_cost to 0.1
        network.storage_units.loc[storage_name, 'spill_cost'] = 0.1

        network.storage_units.loc[storage_name, 'efficiency_store'] = 0.866025 #use phs efficiency (hydro didnt have an efficiency, but i want to model them all as the same)

        # Fix unrealistic max_hours values
        current_max_hours = network.storage_units.loc[storage_name, 'max_hours']

        if 'PHS' in storage_name:
            # PHS with missing data - set to typical range
            network.storage_units.loc[storage_name, 'max_hours'] = 8.0
            print(f"Fixed {storage_name}: set max_hours to 8.0")

        elif 'hydro' in storage_name:
            # Hydro with unrealistic data - set to validated range
            network.storage_units.loc[storage_name, 'max_hours'] = 6.0
            print(f"Fixed {storage_name}: corrected max_hours from {current_max_hours} to 6.0")


    fix_artificial_lines_reasonable(network)
    remove_offshore_wind(network)

    return network

# Load your network
network_file_path= "/Users/antoniagrindrod/Documents/pypsa-earth_project/pypsa-earth-RL/networks/elec_s_10_ec_lc1.0_1h.nc"
network = create_pypsa_network(network_file_path)

# Optimize with Gurobi (using your valid license)
network.optimize(solver_name='gurobi')

# Calculate total objective value
total_objective = 0.0
for snapshot in network.snapshots.indices:
    snapshot_weighting = network.snapshot_weightings.objective.loc[snapshot]
    
    # Generator operational costs
    if len(network.generators) > 0:
        gen_marginal_costs = network.generators['marginal_cost']
        gen_power = network.generators_t.p.loc[snapshot]
        gen_cost = (gen_marginal_costs * gen_power).sum()
        total_objective += gen_cost * snapshot_weighting
    
    # Storage unit operational costs
    if len(network.storage_units) > 0:
        # Dispatch costs
        storage_marginal_costs = network.storage_units['marginal_cost']
        storage_p_dispatch = network.storage_units_t.p_dispatch.loc[snapshot]
        storage_cost = (storage_marginal_costs * storage_p_dispatch).sum()
        total_objective += storage_cost * snapshot_weighting
        
        # Storage costs
        storage_marginal_costs_storage = network.storage_units['marginal_cost_storage']
        storage_store_power = network.storage_units_t.p_store.loc[snapshot]
        storage_store_cost = (storage_marginal_costs_storage * storage_store_power).sum()
        total_objective += storage_store_cost * snapshot_weighting
        
        # Spill costs
        if hasattr(network.storage_units_t, 'spill'):
            spill_costs = network.storage_units['spill_cost']
            spill_amounts = network.storage_units_t.spill.loc[snapshot]
            spill_cost = (spill_costs * spill_amounts).sum()
            total_objective += spill_cost * snapshot_weighting

optimization_file_path = '/Users/antoniagrindrod/Documents/pypsa-earth_project/pypsa-earth-RL/RL/optimized_network/elec_s_10_ec_lc1.0_1h_Test_Objective.txt'

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(optimization_file_path), exist_ok=True)

# Save the float value
with open(optimization_file_path, 'w') as f:
    f.write(str(total_objective))

print(f"Optimization result saved: {total_objective}")