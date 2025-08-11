import pypsa
import numpy as np
import gc
import pandas as pd
import os
import pickle
from google.colab import drive

def _variable_constraint_mapping(network_file, constraints_to_skip):
    """
    Initialize all optimization components in one pass to avoid creating multiple models.
    This method:
    1. Creates the optimization model once
    2. Extracts objective components (vars, coeffs, const)
    3. Creates the variable ID to name mapping
    4. Extracts constraints
    5. Cleans up the model
    """
    network = pypsa.Network(network_file)
    # Create model once - this is an expensive operation
    temp_model = network.optimize.create_model()

    # Create variable ID to name mapping
    var_id_to_name = {}
    for var_name, variable in temp_model.variables.items():
        # Get the variable labels (IDs) for this variable
        var_labels = variable.labels

        if hasattr(var_labels, 'values'):
            # Multi-dimensional variable
            labels_flat = var_labels.values.flatten()
            coords = variable.labels.coords
            for i, label in enumerate(labels_flat):
                if label != -1:  # -1 means no variable
                    # Create a name that includes the index for multi-dim variables
                    if len(coords) > 0:
                        # Get the coordinate values for this flat index
                        unravel_idx = np.unravel_index(i, var_labels.shape)
                        coord_values = []
                        for dim_idx, dim_name in enumerate(var_labels.dims):
                            coord_val = coords[dim_name].values[unravel_idx[dim_idx]]

                            # Handle datetime64 values properly
                            if isinstance(coord_val, np.datetime64) or hasattr(coord_val, 'strftime'):
                                # Convert datetime to string in ISO format
                                try:
                                    coord_val = pd.Timestamp(coord_val).isoformat()
                                except:
                                    # Fallback if conversion fails
                                    coord_val = str(coord_val)

                            coord_values.append(f"{dim_name}={coord_val}")

                        full_name = f"{var_name}[{','.join(coord_values)}]"
                    else:
                        full_name = f"{var_name}[{i}]"
                    var_id_to_name[label] = full_name
        else:
            # Scalar variable
            var_id_to_name[var_labels] = var_name

        # Store constraint information
        constraints = {}
        for name, constraint_group in temp_model.constraints.items():
            # Corrected condition to skip desired constraints
            if name in constraints_to_skip:
              continue
            # Check if this is a constraint group with multiple individual constraints
            if hasattr(constraint_group.lhs, 'shape') and len(constraint_group.lhs.shape) > 0:
                # This is a constraint group with multiple individual constraints
                # We need to extract each individual constraint

                # Get the dimensions of the constraint group
                dims = constraint_group.lhs.dims if hasattr(constraint_group.lhs, 'dims') else []

                # If it has dimensions, iterate through each individual constraint
                if dims:
                    # Get coordinate values for each dimension
                    coords = {}
                    for dim in dims:
                        if hasattr(constraint_group.lhs, 'coords') and dim in constraint_group.lhs.coords:
                            coords[dim] = constraint_group.lhs.coords[dim].values

                    # Create a flat iterator through all combinations of coordinates
                    if coords:
                        try:
                            # Create all combinations of coordinate indices - only use dimensions that exist in coords
                            valid_dims = [dim for dim in dims if dim in coords]
                            if not valid_dims:
                                # No valid dimensions found, skip this constraint group
                                print(f"Warning: No valid dimensions found for constraint {name}")
                                continue

                            # Create shape tuple for ndindex
                            shape_tuple = tuple(len(coords[dim]) for dim in valid_dims)
                            if not shape_tuple:
                                # Empty shape tuple, skip this constraint group
                                print(f"Warning: Empty shape tuple for constraint {name}")
                                continue

                            # Create iterator
                            indices = np.ndindex(shape_tuple)

                            # Iterate through all combinations
                            for idx in indices:
                                try:
                                    # Create a key for this specific constraint
                                    coord_values = []
                                    for i, dim in enumerate(valid_dims):
                                        coord_values.append(f"{dim}={coords[dim][idx[i]]}")

                                    specific_key = f"{name}[{','.join(coord_values)}]"

                                    # Extract the specific constraint values - with error handling
                                    try:
                                        # For LHS
                                        if hasattr(constraint_group.lhs.vars, '__getitem__') and hasattr(constraint_group.lhs, 'coeffs'):
                                            # Create a proper index for this specific constraint
                                            # We need to map our valid_dims indices to the full dims indices

                                            # For linear expressions - safely get values
                                            try:
                                                if hasattr(constraint_group.lhs.vars, '__getitem__'):
                                                    lhs_vars = constraint_group.lhs.vars[idx]
                                                    #condition is evaluating to true and type(constraint_group.lhs.vars) is <class 'xarray.core.dataarray.DataArray'>
                                                else:
                                                    lhs_vars = constraint_group.lhs.vars

                                                if hasattr(constraint_group.lhs.coeffs, '__getitem__'):
                                                    lhs_coeffs = constraint_group.lhs.coeffs[idx]
                                                else:
                                                    lhs_coeffs = constraint_group.lhs.coeffs
                                            except Exception as e:
                                                print(f"Warning: Error accessing constraint values for {specific_key}: {e}")
                                                continue

                                            # Create a new linear expression for this specific constraint
                                            specific_lhs = type('LinearExpr', (), {
                                                'vars': np.array([[lhs_vars]]) if np.isscalar(lhs_vars) else np.array([lhs_vars]),
                                                'coeffs': np.array([[lhs_coeffs]]) if np.isscalar(lhs_coeffs) else np.array([lhs_coeffs]),
                                                'copy': lambda self: self
                                            })

                                            # Add constant if it exists - safely
                                            if hasattr(constraint_group.lhs, 'const'):
                                                try:
                                                    if hasattr(constraint_group.lhs.const, '__getitem__'):
                                                        const_val = constraint_group.lhs.const[idx]
                                                    else:
                                                        const_val = constraint_group.lhs.const
                                                    specific_lhs.const = np.array([[const_val]]) if np.isscalar(const_val) else np.array([const_val])
                                                except Exception as e:
                                                    # If error accessing const, just use 0
                                                    specific_lhs.const = 0
                                                    print(f"Warning: Error accessing const for {specific_key}: {e}")
                                        # else:
                                        #     # For simple values
                                        #     try:
                                        #         if hasattr(constraint_group.lhs, '__getitem__'):
                                        #             specific_lhs = constraint_group.lhs[idx]
                                        #         else:
                                        #             specific_lhs = constraint_group.lhs
                                        #     except Exception as e:
                                        #         print(f"Warning: Error accessing LHS for {specific_key}: {e}")
                                        #         continue

                                        # For RHS - safely
                                        try:
                                            if hasattr(constraint_group.rhs, '__getitem__'):
                                                rhs_val = constraint_group.rhs[idx]
                                            else:
                                                rhs_val = constraint_group.rhs
                                            specific_rhs = np.array([[rhs_val]]) if np.isscalar(rhs_val) else np.array([rhs_val])
                                        except Exception as e:
                                            print(f"Warning: Error accessing RHS for {specific_key}: {e}")
                                            continue

                                        # For sign - safely
                                        try:
                                            if hasattr(constraint_group.sign, '__getitem__'):
                                                sign_val = constraint_group.sign[idx].values.item()
                                            else:
                                                sign_val = constraint_group.sign
                                            specific_sign = np.array([sign_val]) if np.isscalar(sign_val) else np.array([sign_val])
                                        except Exception as e:
                                            print(f"Warning: Error accessing sign for {specific_key}: {e}")
                                            specific_sign = np.array(['>=']) # Default sign

                                        # Store this specific constraint
                                        constraints[specific_key] = {
                                            'lhs': specific_lhs,
                                            'rhs': specific_rhs,
                                            'sign': specific_sign
                                        }
                                    except Exception as e:
                                        print(f"Warning: Error processing constraint {specific_key}: {e}")
                                        continue
                                except Exception as e:
                                    print(f"Warning: Error creating key for constraint: {e}")
                                    continue
                        except Exception as e:
                            print(f"Warning: Error creating indices for constraint {name}: {e}")
                            continue
                else: #no case handling for when no dimensions but still has shape
                    print(f"Warning: No dimensions found for constraint {name}")
                    continue
            else:
                # This is a single constraint, store it directly
                try:
                    constraints[name] = {
                        'lhs': constraint_group.lhs.copy() if hasattr(constraint_group.lhs, 'copy') else constraint_group.lhs,
                        'rhs': constraint_group.rhs.copy() if hasattr(constraint_group.rhs, 'copy') else constraint_group.rhs,
                        'sign': constraint_group.sign
                    }
                except Exception as e:
                    print(f"Warning: Error storing single constraint {name}: {e}")
                    continue

        # Clean up to free memory
        del temp_model
        gc.collect()

        return var_id_to_name, constraints

def save_mappings(var_id_to_name, constraints, network_file, output_dir="var_constraint_map"):
    """
    Save the variable ID to name mapping and constraints to files in Google Drive.
    
    Parameters:
    -----------
    var_id_to_name : dict
        Mapping from variable IDs to variable names
    constraints : dict
        Dictionary of constraints
    network_file : str
        Path to the network file used to create the mappings
    output_dir : str, optional
        Directory relative to Google Drive MyDrive to save the mapping files to
        
    Returns:
    --------
    tuple : (var_map_path, constraints_path) - Paths to the saved mapping files
    """
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Create the full path to Google Drive directory
    gdrive_base = '/content/drive/MyDrive/Colab_Notebooks'
    full_output_dir = os.path.join(gdrive_base, output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Get the network filename without path or extension
    network_name = os.path.basename(network_file)
    network_name = os.path.splitext(network_name)[0]
    
    # Create output filenames
    var_map_file = os.path.join(full_output_dir, f"{network_name}_var_id_to_name.pkl")
    constraints_file = os.path.join(full_output_dir, f"{network_name}_constraints.pkl")
    
    # Save mappings to files
    with open(var_map_file, 'wb') as f:
        pickle.dump(var_id_to_name, f)
        
    with open(constraints_file, 'wb') as f:
        pickle.dump(constraints, f)
    
    print(f"Saved variable mapping to: {var_map_file}")
    print(f"Saved constraints to: {constraints_file}")
    
    return var_map_file, constraints_file

def load_mappings(network_file, input_dir="var_constraint_map"):
    """
    Load previously saved variable ID to name mapping and constraints from files in Google Drive.
    
    Parameters:
    -----------
    network_file : str
        Path to the network file used to create the mappings
    input_dir : str, optional
        Directory relative to Google Drive MyDrive where the mapping files are stored
        
    Returns:
    --------
    tuple : (var_id_to_name, constraints) - The loaded mappings
    """
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Create the full path to Google Drive directory
    gdrive_base = '/content/drive/MyDrive/Colab_Notebooks'
    full_input_dir = os.path.join(gdrive_base, input_dir)
    
    # Get the network filename without path or extension
    network_name = os.path.basename(network_file)
    network_name = os.path.splitext(network_name)[0]
    
    # Create input filenames
    var_map_file = os.path.join(full_input_dir, f"{network_name}_var_id_to_name.pkl")
    constraints_file = os.path.join(full_input_dir, f"{network_name}_constraints.pkl")
    
    # Check if files exist
    if not os.path.exists(var_map_file) or not os.path.exists(constraints_file):
        raise FileNotFoundError(f"Mapping files for {network_name} not found in {full_input_dir}")
    
    # Load mappings from files
    with open(var_map_file, 'rb') as f:
        var_id_to_name = pickle.load(f)
        
    with open(constraints_file, 'rb') as f:
        constraints = pickle.load(f)
    
    print(f"Loaded variable mapping from: {var_map_file}")
    print(f"Loaded constraints from: {constraints_file}")
    
    return var_id_to_name, constraints

if __name__ == "__main__":
    # Example usage
    network_file = "networks/elec_s_5_ec_lc1.0_3h.nc"
    constraints_to_skip = [
        "StorageUnit-fix-p_dispatch-lower", 
        "StorageUnit-fix-p_dispatch-upper", 
        "StorageUnit-fix-p_store-lower", 
        "StorageUnit-fix-p_store-upper", 
        "StorageUnit-fix-state_of_charge-lower", 
        "StorageUnit-fix-state_of_charge-upper",
        "StorageUnit-energy_balance"
    ]
    
    # Create mappings
    var_id_to_name, constraints = _variable_constraint_mapping(network_file, constraints_to_skip)
    
    # Save mappings
    var_map_file, constraints_file = save_mappings(var_id_to_name, constraints, network_file)
    
    print(f"Created variable ID to name mapping with {len(var_id_to_name)} entries")
    print(f"Created constraints mapping with {len(constraints)} entries")
    print(f"Saved variable mapping to {var_map_file}")
    print(f"Saved constraints mapping to {constraints_file}")


