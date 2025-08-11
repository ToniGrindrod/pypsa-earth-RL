# Variable and Constraint Mapping for PyPSA Networks

This directory stores cached mappings of variable IDs to variable names and constraints for PyPSA networks.

## Purpose

When working with PyPSA networks in reinforcement learning environments, we often need to:

1. Map variable IDs to their names for objective function evaluation
2. Extract constraints for constraint evaluation

Creating these mappings is computationally expensive as it requires creating a full optimization model. By caching these mappings, we can:

- Significantly reduce initialization time for RL environments
- Separate mapping creation from environment initialization
- Reuse mappings across multiple runs

## Usage

To use these mappings in your code:

```python
from RL.variable_constraint_mappinge import get_or_create_mappings

# Get mappings for a network, creating and caching them if they don't exist
var_id_to_name, constraints = get_or_create_mappings(
    network_file="path/to/network.nc",
    constraints_to_skip=["StorageUnit-energy_balance"],  # Optional
    cache_dir="RL/var_constraint_map"  # Optional
)

# Use the mappings in your environment
# ...
```

## File Naming Convention

Files in this directory follow the naming convention:

- `{network_name}_var_id_to_name.pkl`: Variable ID to name mapping
- `{network_name}_constraints.pkl`: Constraints mapping

Where `{network_name}` is the filename of the network without path or extension.

## Implementation Details

The mappings are created using the `_variable_constraint_mapping` function in `RL/variable_constraint_mappinge.py`. This function:

1. Creates a PyPSA optimization model
2. Extracts variable IDs and their names
3. Extracts constraints, filtering out any specified in `constraints_to_skip`
4. Saves the mappings to pickle files for future use

See `RL/var_constraint_map_example.py` for a complete example of how to use these mappings.
