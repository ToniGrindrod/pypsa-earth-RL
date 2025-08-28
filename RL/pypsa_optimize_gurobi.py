import pypsa

def fix_artificial_lines_hardcoded(network):
    """
    Fix artificial lines with hardcoded values to match existing non-extendable lines:
    - s_nom = 442.4 (your desired capacity)
    - s_nom_extendable = False
    - s_nom_min = 0
    - s_nom_max = inf
    - extendable = False (if column exists)
    """
    print("=== FIXING ARTIFICIAL LINES WITH HARDCODED VALUES ===")
    fixed_capacity=442.4

    # Find artificial lines
    artificial_lines = [line for line in network.lines.index
                       if any(keyword in str(line).lower() for keyword in ['new', '<->', 'artificial'])]

    print(f"Found {len(artificial_lines)} artificial lines to fix:")

    # Fix each artificial line with hardcoded values
    for line_name in artificial_lines:
        print(f"\n🔧 Fixing: {line_name}")

        # s_nom = 442.4 (your fixed capacity)
        old_s_nom = network.lines.loc[line_name, 's_nom']
        network.lines.loc[line_name, 's_nom'] = fixed_capacity
        print(f"    s_nom: {old_s_nom} → {fixed_capacity}")

        # s_nom_extendable = False
        if 's_nom_extendable' not in network.lines.columns:
            network.lines['s_nom_extendable'] = False
        network.lines.loc[line_name, 's_nom_extendable'] = False
        print(f"    s_nom_extendable: → False")

        # s_nom_min = 0
        if 's_nom_min' not in network.lines.columns:
            network.lines['s_nom_min'] = 0.0
        network.lines.loc[line_name, 's_nom_min'] = 0.0
        print(f"    s_nom_min: → 0.0")

        # s_nom_max = inf
        if 's_nom_max' not in network.lines.columns:
            network.lines['s_nom_max'] = float('inf')
        network.lines.loc[line_name, 's_nom_max'] = float('inf')
        print(f"    s_nom_max: → inf")

    return network

def create_pypsa_network(network_file):
    """Create a PyPSA network from the .nc file."""
    # Initialize network
    network = pypsa.Network(network_file)
    for storage_name in network.storage_units.index:
        # Use .loc for direct assignment to avoid SettingWithCopyWarning
        network.storage_units.loc[storage_name, 'cyclic_state_of_charge'] = False

    fix_artificial_lines_hardcoded(network)

    return network

# Load your network
network_file_path= "/Users/antoniagrindrod/Documents/pypsa-earth_project/pypsa-earth-RL/networks/elec_s_5_ec_lc1.0_3h.nc"
network = create_pypsa_network(network_file_path)

# Optimize with Gurobi (using your valid license)
network.optimize(solver_name='gurobi')


# Save the optimized network
network.export_to_netcdf("/Users/antoniagrindrod/Desktop/optimized_elec_s_5_ec_lc1.0_3h.nc")
print("Optimization complete and saved!")