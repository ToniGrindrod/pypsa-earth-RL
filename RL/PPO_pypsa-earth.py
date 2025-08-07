import pypsa
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import gc
import psutil
import matplotlib.pyplot as plt

import neptune

from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

def create_pypsa_network(network_file):
    """Create a PyPSA network from the .nc file."""
    # Initialize network
    n = pypsa.Network(network_file)

    return n

def calculate_offset_k_initialization(envClass,k_method='mean', k_samples=1000):
   """
   Calculate the offset k for replacement reward method.
   This ensures valid states always have higher rewards than invalid states.

   Parameters:
   -----------
   k_samples : int
       Number of random samples to use for estimation
   k_method : str
       Method to calculate k: 'mean' or 'worst_case'

   Returns:
   --------
   float: Offset value k
   """
   print(f"Sampling {k_samples} random states to calculate offset k...")

   temp_env = envClass()
   #this initializes episode_length to number of snapshots and constraint_penalty_factor to None.
   #I'm just making this env to access certain attributes/ methods; which should be fine since none of these attributes/methods reference these two parameters.

   objective_values = []
   successful_samples = 0

   try:
       for i in range(k_samples):
           try:
               # STEP 1: Sample ONE random snapshot (random state)
               random_snapshot_idx = np.random.randint(0, temp_env.total_snapshots)

               # STEP 2: Sample ONE random action for that state
               random_action = np.random.random(temp_env.n_non_slack)
               scaled_action = temp_env.scale_action(random_action)

               # STEP 3: Apply the action to the sampled state
               for j, gen_name in enumerate(temp_env.non_slack_names):
                   temp_env.network.generators_t.p_set.iloc[random_snapshot_idx,
                       temp_env.network.generators_t.p_set.columns.get_loc(gen_name)] = scaled_action[j]

               # STEP 4: Evaluate objective for this ONE state-action combination
               temp_env.snapshot_idx = random_snapshot_idx  # Set snapshot for evaluation
               temp_env.network.lpf()
               obj_value = temp_env._evaluate_stored_objective()
               objective_values.append(obj_value)
               successful_samples += 1

               # Progress indicator every 200 samples
               if (i + 1) % 200 == 0:
                   print(f"  Completed {i + 1}/{k_samples} samples...")

           except Exception as e:
               # Skip failed samples but continue
               if i < 5:  # Only print first few errors to avoid spam
                   print(f"  Sample {i} failed: {e}")
               continue

       # Calculate offset based on method
       if objective_values:
           if k_method == 'worst_case':
               k = abs(max(objective_values))
               print(f"  Using worst-case method: k = |{max(objective_values):.2f}| = {k:.2f}")
           else:  # method == 'mean'
               mean_val = np.mean(objective_values)
               k = abs(mean_val)
               print(f"  Using mean method: k = |{mean_val:.2f}| = {k:.2f}")

           print(f"  Successfully sampled {successful_samples}/{k_samples} states")
           print(f"  Objective value range: [{min(objective_values):.2f}, {max(objective_values):.2f}]")
       else:
           print("  Warning: No successful samples, using default k value")
           k = 2500  # Default fallback value

   finally:
       # Clean up temporary environment (optional but good practice)
       del temp_env

   return k

def get_variable_value(network,var_name):
        """
        Get the current value of a single optimization variable from the network.

        Parameters:
        -----------
        network : pypsa.Network
            The PyPSA network object
        var_name : str
            Variable name like 'Generator-p[snapshot=now,Generator=coal_gen_1]'

        Returns:
        --------
        float : current value of the variable
        """
        # Parse the variable name
        # Format: ComponentName-attribute[dimension1=value1,dimension2=value2,...]

        # Split on first '[' to separate base name from coordinates
        if '[' in var_name:
            base_name, coords_str = var_name.split('[', 1)
            coords_str = coords_str.rstrip(']')  # Remove trailing ']'
        else:
            base_name = var_name
            coords_str = ""

        # Parse base name: ComponentName-attribute
        component_name, attribute = base_name.split('-', 1)

        # Parse coordinates if they exist
        coords = {}
        if coords_str:
            # Split by comma and parse key=value pairs
            for coord_pair in coords_str.split(','):
                key, value = coord_pair.split('=', 1)
                coords[key.strip()] = value.strip()

        # Determine if this has time dimension (snapshot)
        has_snapshot = 'snapshot' in coords

        if has_snapshot:
            # Access dynamic dataframe using n.dynamic()
            snapshot_value = coords['snapshot']
            component_instance = coords[component_name]

            # Special handling for branch flow variables
            if component_name in network.passive_branch_components and attribute == 's':
                # For branch components, 's' is stored as 'p0' in the network
                # We can use p0 directly as the value of 's'
                try:
                    return network.dynamic(component_name)['p0'].loc[snapshot_value, component_instance]
                except Exception as e:
                    print(f"Warning: Could not get flow value for {var_name}: {e}")
                    return 0.0

            # Get dynamic data for normal case
            dynamic_data = network.dynamic(component_name)

            # Access the specific attribute DataFrame
            if attribute in dynamic_data:
                return dynamic_data[attribute].loc[snapshot_value, component_instance]
            else:
                raise KeyError(f"Attribute {attribute} not found in dynamic data for {component_name}")

        else:
            # Access static dataframe using n.static()
            component_instance = coords[component_name]

            # Get static data
            static_data = network.static(component_name)

            # Access the value
            return static_data.loc[component_instance, attribute]

def create_variable_values_mapping(network, variable_names):
        """
        Create a mapping from optimization variable names to their current values in the network.

        Parameters:
        -----------
        network : pypsa.Network
            The PyPSA network object
        variable_names : list
            List of variable names like ['Generator-p[snapshot=now,Generator=coal_gen_1]', ...]

        Returns:
        --------
        dict : mapping from variable name to current value
        """
        var_values = {}

        for var_name in variable_names:
            try:
                value = get_variable_value(network,var_name)
                var_values[var_name] = value
            except Exception as e:
                print(f"Warning: Could not get value for {var_name}: {e}")
                var_values[var_name] = 0.0  # Default fallback

        return var_values

def evaluate_objective(var_id_to_name, network, snapshot_idx):
        """
        Direct evaluation without mock objects.
        Only includes terms from the current snapshot in the objective function.
        """
        temp_model = network.optimize.create_model()

        # Extract objective components
        obj_expr = temp_model.objective
        objective_vars = obj_expr.vars.copy()
        objective_coeffs = obj_expr.coeffs.copy()
        objective_const = obj_expr.const.copy() if hasattr(obj_expr, 'const') else 0

        # Get variable name mapping for current network
        id_to_name = var_id_to_name

        # Get the current snapshot name
        current_snapshot = snapshot_idx

        # Get current variable values
        variable_names = []
        var_indices = []
        vars_flat = objective_vars.values.flatten()
        coeffs_flat = objective_coeffs.values.flatten()

        # Filter variables to only include those from the current snapshot
        for i, var_id in enumerate(vars_flat):
            if var_id != -1 and var_id in id_to_name:
                var_name = id_to_name[var_id]
                # Check if this variable belongs to the current snapshot
                if 'snapshot=' in var_name:
                    # Extract the snapshot value from the variable name
                    snapshot_part = var_name.split('snapshot=')[1].split(',')[0].split(']')[0]
                    if snapshot_part == str(current_snapshot):
                        variable_names.append(var_name)
                        var_indices.append(i)
                else:
                    # Include variables without snapshot dimension (like investment variables)
                    variable_names.append(var_name)
                    var_indices.append(i)

        var_values = create_variable_values_mapping(network,variable_names)

        # Direct mathematical evaluation using only variables from current snapshot
        if var_indices:
            result = np.sum(coeffs_flat[var_indices] *
                        [var_values.get(name, 0) for name in variable_names]) + \
                    objective_const
        else:
            # If no variables for this snapshot, just return the constant
            result = objective_const

        return result

def evaluate_baseline_reward(network_file, env, agent):
    network_baseline=create_pypsa_network(network_file)#network to run pypsa optimize on
    network_baseline.optimize()
    snapshots=network_baseline.snapshots
    objective_sum=0
    for snapshot_idx in range(len(snapshots)):
        objective_sum+=evaluate_objective(env.var_id_to_name, network_baseline, snapshots[snapshot_idx])
    baseline_reward_value=-objective_sum*(env.episode_length/ len(snapshots))#assuming episode length is a multiple of the number of snapshots
    #reward is -1 times the objective value
    return baseline_reward_value

class Env2Gen1LoadConstr(gym.Env):
    """
    OpenAI Gym environment for Optimal Power Flow using PyPSA.
    Simple example with 2 generators and 1 load.
    Implemented without constraints (omitted bus voltage limits, line limits, etc.).

    Action Space: Continuous setpoints for generators within their capacity limits
    Observation Space: Network state, represented by:
    - the active and reactive power of all loads,
    - the reactive power prices of all generators,
    - the active power setpoints of all generators
    (This follows http://arxiv.org/abs/2403.17831. Additional variables can be added optionally)
    """

    def __init__(self,network_file, episode_length=None, constraint_penalty_factor=100):
        super(Env2Gen1LoadConstr, self).__init__()

        # Use provided network or create new one
        self.network =create_pypsa_network(network_file)

        self._initialize_optimization_components()
        self.penalty_factor=constraint_penalty_factor

        # Episode management
        self.total_snapshots = len(self.network.snapshots)
        self.episode_length = episode_length if episode_length is not None else self.total_snapshots
        self.current_step = 0  # Steps within current episode
        self.snapshot_idx = 0  # Current snapshot index (cycles through all snapshots)

        # Find the slack generator
        # The agent will only be able to control the active power setpoints of generators that are not the slack generator
        self.slack_generator_idx = self.network.generators[self.network.generators.control == "Slack"].index
        #Note that there should only ever be one slack bus.
        #Need to implement handling for multiple generators control set to "Slack",
        #In that case, will need to designate one for slack and change control of others
        #TO DO: Implement handling for multiple slack generators

        #Get indices of all generators that are not the slack generator (after ensuring there is only one slack generator)
        self.non_slack_gens = self.network.generators[self.network.generators.control != "Slack"].index #must explicitly search again, can't reference results of last query since this may have changed since choosing slack generator
        self.non_slack_names = list(self.non_slack_gens)
        self.n_non_slack = len(self.non_slack_names)
        self.non_slack_df = self.network.generators.loc[self.non_slack_gens]  # Fixed: was non_slack_generators
        # The agent will only be able to control the active power setpoints of generators that are not the slack generator

        # Get generator limits (in MW)
        self.a_min = (self.non_slack_df.p_min_pu * self.non_slack_df.p_nom).values
        self.a_max = (self.non_slack_df.p_max_pu * self.non_slack_df.p_nom).values
        #TO DO: When generalize, control of the generator will determine the action space of the agent
        # Whichever quantity is controllable, according to the control, will be the quantity the agent chooses as its action (e.g. could be active or reactive power)
        # Then we would need to change the bounds of the the relevant bounds will be the actions space to be the min and max of whatever the quantity is

        # Define action space: continuous setpoints for each generator within limits
        # Action space is a Box with shape (n_non_slack,) where each element has value between 0 and 1.
        self.action_space = gym.spaces.Box(0, 1, shape=(self.n_non_slack,))

        # Define observation space - network state
        # This will include: generator outputs, loads, voltages, line flows, etc.
        # For simplicity, let's include key network variables
        # obs_dim = (self.n_non_slack+1)+len(self.network.loads)
        # First term: self.n_non_slack: Active power outputs of all non-slackgenerators! Actually I removed this because that actually corresponds to the action choice.
        # TO DO: Later we might need to include a history of the network as part of the observation - to obey ramp constraints etc.
        #Second term: Cost function components (start with one each - marginal cost) of all generators, including slack
        # Third term: Active power of load demands
        # TO DO: include power of storage units and noncontrollable generators (small generators and storage units; not including slack generator) in observation space


        #TO DO: If use .pf instead of .lpf, add another of each term for active power AND reactive power
        #include the power of the slack generator in the observation, since agent's action (choice of active power for non-slack generators influences power of slack generator, through power flow calculations)

        # Initialize the network state
        self.reset()

        # Observation space is bounded - you may want to adjust these bounds based on your system

        # Create observation space
        low_bounds, high_bounds = self.create_observation_bounds()
        self.observation_space = spaces.Box(
            low=low_bounds,
            high=high_bounds,
            dtype=np.float32
        )

        #Bounds are set according to specific component of the observation
        #Could set the bounds of active power of each generator to correspond to min and max p

    def _initialize_optimization_components(self):
        """
        Initialize all optimization components in one pass to avoid creating multiple models.
        This method:
        1. Creates the optimization model once
        2. Extracts objective components (vars, coeffs, const)
        3. Creates the variable ID to name mapping
        4. Extracts constraints
        5. Cleans up the model
        """
        # Create model once - this is an expensive operation
        temp_model = self.network.optimize.create_model()

        # Extract objective components
        obj_expr = temp_model.objective
        self.objective_vars = obj_expr.vars.copy()
        self.objective_coeffs = obj_expr.coeffs.copy()
        self.objective_const = obj_expr.const.copy() if hasattr(obj_expr, 'const') else 0

        # Create variable ID to name mapping
        self.var_id_to_name = {}
        for var_name, variable in temp_model.variables.items():
            # Get the variable labels (IDs) for this variable
            var_labels = variable.labels

            if hasattr(var_labels, 'values'):
                # Multi-dimensional variable
                labels_flat = var_labels.values.flatten()
                for i, label in enumerate(labels_flat):
                    if label != -1:  # -1 means no variable
                        # Create a name that includes the index for multi-dim variables
                        coords = variable.labels.coords
                        if len(coords) > 0:
                            # Get the coordinate values for this flat index
                            unravel_idx = np.unravel_index(i, var_labels.shape)
                            coord_values = []
                            for dim_idx, dim_name in enumerate(var_labels.dims):
                                coord_val = coords[dim_name].values[unravel_idx[dim_idx]]
                                coord_values.append(f"{dim_name}={coord_val}")
                            full_name = f"{var_name}[{','.join(coord_values)}]"
                        else:
                            full_name = f"{var_name}[{i}]"
                        self.var_id_to_name[label] = full_name
            else:
                # Scalar variable
                self.var_id_to_name[var_labels] = var_name

        # Store constraint information
        self.constraints = {}
        for name, constraint_group in temp_model.constraints.items():
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
                                        if hasattr(constraint_group.lhs, 'vars') and hasattr(constraint_group.lhs, 'coeffs'):
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
                                        self.constraints[specific_key] = {
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
                    self.constraints[name] = {
                        'lhs': constraint_group.lhs.copy() if hasattr(constraint_group.lhs, 'copy') else constraint_group.lhs,
                        'rhs': constraint_group.rhs.copy() if hasattr(constraint_group.rhs, 'copy') else constraint_group.rhs,
                        'sign': constraint_group.sign
                    }
                except Exception as e:
                    print(f"Warning: Error storing single constraint {name}: {e}")
                    continue

        # Clean up to free memory
        del temp_model, obj_expr
        gc.collect()

    def create_observation_bounds(self):
        """
        Create bounds for the observation space based on generator costs and load power
        """
        # --- Generator Cost Bounds ---
        gen_cost_lower = []
        gen_cost_upper = []

        for gen in self.network.generators.index:
            # Get generator specs
            p_nom = self.network.generators.loc[gen, "p_nom"]
            p_min_pu = self.network.generators.loc[gen, "p_min_pu"]
            p_max_pu = self.network.generators.loc[gen, "p_max_pu"]
            marginal_cost = self.network.generators.loc[gen, "marginal_cost"]

            # Min cost: generator at minimum output
            min_cost = p_min_pu * p_nom * marginal_cost

            # Max cost: generator at maximum output
            max_cost = p_max_pu * p_nom * marginal_cost
            #TO DO: add other cost function components if applicable
            gen_cost_lower.append(min_cost)
            gen_cost_upper.append(max_cost)

        # --- Load Power Bounds ---
        load_p_lower = []
        load_p_upper = []

        for load in self.network.loads.index:
            # Get load specs
            p_nom = self.network.loads.loc[load, "p_set"]#TO DO: not sure if this is correct. in general might not set p_set when specifying load component (might instead specifiy p_nom)

            # If you have time series data for loads
            if hasattr(self.network, "loads_t") and not self.network.loads_t.p.empty:
                min_load = self.network.loads_t.p.iloc[self.snapshot_idx].min()
                max_load = self.network.loads_t.p.iloc[self.snapshot_idx].max()
            else:
                # If no time series, use p_set with some margin
                min_load = 0.7 * p_nom  # Assumption: load can go down to 70%
                max_load = 1.3 * p_nom  # Assumption: load can go up to 130%

            load_p_lower.append(min_load)
            load_p_upper.append(max_load)

        # Combine all bounds
        low_bounds = np.concatenate([gen_cost_lower, load_p_lower]).astype(np.float32)
        high_bounds = np.concatenate([gen_cost_upper, load_p_upper]).astype(np.float32)

        return low_bounds, high_bounds

    def reset_network(self):
        """Reset and ensure essential DataFrames exist."""
        #Note that we do not just create a new network here, as this consumes more memory and previously led to a segmentation fault
        # TO DO: For a general network, we would need to reset all time-varying data (i.e. all components with time-varying data)
        # Ensure generators_t DataFrames exist and are reset
        if not hasattr(self.network, 'generators_t'):
            # This shouldn't happen with PyPSA, but just in case
            pass

        # Initialize/reset generators_t.p
        if not hasattr(self.network.generators_t, 'p') or self.network.generators_t.p.empty:
            self.network.generators_t.p = pd.DataFrame(
                0.0,
                index=self.network.snapshots,
                columns=self.network.generators.index
            )
        else:
            self.network.generators_t.p.iloc[:, :] = 0.0

        # Initialize/reset generators_t.p_set
        if not hasattr(self.network.generators_t, 'p_set') or self.network.generators_t.p_set.empty:
            self.network.generators_t.p_set = pd.DataFrame(
                0.0,
                index=self.network.snapshots,
                columns=self.network.generators.index
            )
        else:
            self.network.generators_t.p_set.iloc[:, :] = 0.0

        # Initialize/reset loads_t.p based on loads_t.p_set
        if not hasattr(self.network.loads_t, 'p') or self.network.loads_t.p.empty:
            # Create loads_t.p dataframe with same structure as loads_t.p_set
            self.network.loads_t.p = self.network.loads_t.p_set.copy()
        else:
            # Reset loads_t.p to match loads_t.p_set
            self.network.loads_t.p.iloc[:, :] = self.network.loads_t.p_set.values
        # # Reset other existing time-series DataFrames (but don't create new ones)
        # component_types = ['loads', 'buses', 'lines', 'storage_units', 'transformers']

        # for comp_type in component_types:
        #     comp_t_name = f"{comp_type}_t"
        #     if hasattr(self.network, comp_t_name):
        #         comp_t = getattr(self.network, comp_t_name)

        #         for attr_name in dir(comp_t):
        #             if not attr_name.startswith('_'):
        #                 try:
        #                     attr_value = getattr(comp_t, attr_name)
        #                     if isinstance(attr_value, pd.DataFrame) and not attr_value.empty:
        #                         attr_value.iloc[:, :] = 0.0
        #                 except Exception as e:
        #                     pass  # Skip problematic attributes

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        Returns initial observation and info (gymnasium format).
        """
        # Set seed if provided
        if seed is not None:
            self.seed(seed)

        # Reset episode counters
        self.current_step = 0
        self.snapshot_idx = 0

        self.reset_network()
        #self._test_hidden_references()
        #can run that instead to check if there are hidden references preventing network cleanup

        # Get initial observation
        obs = self._get_observation()

        # Info dict for gymnasium compatibility
        info = {
            'current_step': self.current_step,
            'snapshot_idx': self.snapshot_idx
        }

        return obs, info

    def _get_observation(self):
        """
        Get current network state as observation.
        """
        obs_components = []

        # obs_dim = (self.n_non_slack+1)+len(self.network.loads)
        # First term: self.n_non_slack: Active power outputs of all non-slackgenerators! Actually I removed this because that actually corresponds to the action choice.
        # TO DO: Later we might need to include a history of the network as part of the observation - to obey ramp constraints etc.
        # Second term: Cost function components (start with one each - marginal cost) of all generators, including slack
        # Third term: Active power of load demands


        # Generator power costs (for now just marginal) - TO DO: change this to include more components of cost function to make this more accurate
        gen_costs = self.network.generators.marginal_cost.values #TO DO: not sure if the marginal_cost should be accessed from generators_t instead (would do generators.marginal_cost.iloc[snapshot_idx].values). But then not static
        obs_components.append(gen_costs)

        load_demands = np.array([])
        # Load demands - handle case where we might have empty loads_t.p
        if hasattr(self.network, 'loads_t') and not self.network.loads_t.p.empty:
            load_demands = self.network.loads_t.p.iloc[self.snapshot_idx].values
        obs_components.append(load_demands)

        #The following line should be uncommented when want to enforce network constraints via action masking
        #obs_components.append(self._get_mask())

        # TO DO: Could later try to add bus voltages etc. and see if difference on agent learning. (Paper says these are optional components of the observation space)
        # Concatenate all observation components
        observation = np.concatenate(obs_components).astype(np.float32)

        return observation

    def scale_action(self, action):
        """
        Scale action from [0,1] range to [self.a_min, self.a_max] range.

        Parameters:
        -----------
        action : numpy.ndarray
            Action values in range [0,1]

        Returns:
        --------
        numpy.ndarray
            Scaled action in range [self.a_min, self.a_max]
        """
        return self.a_min + action * (self.a_max - self.a_min)

    def get_variable_value(self, var_name):
        """
        Get the current value of a single optimization variable from the network.

        Parameters:
        -----------
        network : pypsa.Network
            The PyPSA network object
        var_name : str
            Variable name like 'Generator-p[snapshot=now,Generator=coal_gen_1]'

        Returns:
        --------
        float : current value of the variable
        """
        # Parse the variable name
        # Format: ComponentName-attribute[dimension1=value1,dimension2=value2,...]

        # Split on first '[' to separate base name from coordinates
        if '[' in var_name:
            base_name, coords_str = var_name.split('[', 1)
            coords_str = coords_str.rstrip(']')  # Remove trailing ']'
        else:
            base_name = var_name
            coords_str = ""

        # Parse base name: ComponentName-attribute
        component_name, attribute = base_name.split('-', 1)

        # Parse coordinates if they exist
        coords = {}
        if coords_str:
            # Split by comma and parse key=value pairs
            for coord_pair in coords_str.split(','):
                key, value = coord_pair.split('=', 1)
                coords[key.strip()] = value.strip()

        # Determine if this has time dimension (snapshot)
        has_snapshot = 'snapshot' in coords

        if has_snapshot:
            # Access dynamic dataframe using n.dynamic()
            snapshot_value = coords['snapshot']
            component_instance = coords[component_name]

            # Special handling for branch flow variables
            if component_name in self.network.passive_branch_components and attribute == 's':
                # For branch components, 's' is stored as 'p0' in the network
                # We can use p0 directly as the value of 's'
                try:
                    return self.network.dynamic(component_name)['p0'].loc[snapshot_value, component_instance]
                except Exception as e:
                    print(f"Warning: Could not get flow value for {var_name}: {e}")
                    return 0.0

            # Get dynamic data for normal case
            dynamic_data = self.network.dynamic(component_name)

            # Access the specific attribute DataFrame
            if attribute in dynamic_data:
                return dynamic_data[attribute].loc[snapshot_value, component_instance]
            else:
                raise KeyError(f"Attribute {attribute} not found in dynamic data for {component_name}")

        else:
            # Access static dataframe using n.static()
            component_instance = coords[component_name]

            # Get static data
            static_data = self.network.static(component_name)

            # Access the value
            return static_data.loc[component_instance, attribute]


    def create_variable_values_mapping(self,variable_names):
        """
        Create a mapping from optimization variable names to their current values in the network.

        Parameters:
        -----------
        network : pypsa.Network
            The PyPSA network object
        variable_names : list
            List of variable names like ['Generator-p[snapshot=now,Generator=coal_gen_1]', ...]

        Returns:
        --------
        dict : mapping from variable name to current value
        """
        var_values = {}

        for var_name in variable_names:
            try:
                value = self.get_variable_value(var_name)
                var_values[var_name] = value
            except Exception as e:
                print(f"Warning: Could not get value for {var_name}: {e}")
                var_values[var_name] = 0.0  # Default fallback

        return var_values

    def _evaluate_stored_objective(self):
        """
        Direct evaluation without mock objects.
        Only includes terms from the current snapshot in the objective function.
        """
        # Get variable name mapping for current network
        id_to_name = self.var_id_to_name

        # Get the current snapshot name
        current_snapshot = self.network.snapshots[self.snapshot_idx]

        # Get current variable values
        variable_names = []
        var_indices = []
        vars_flat = self.objective_vars.values.flatten()
        coeffs_flat = self.objective_coeffs.values.flatten()

        # Filter variables to only include those from the current snapshot
        for i, var_id in enumerate(vars_flat):
            if var_id != -1 and var_id in id_to_name:
                var_name = id_to_name[var_id]
                # Check if this variable belongs to the current snapshot
                if 'snapshot=' in var_name:
                    # Extract the snapshot value from the variable name
                    snapshot_part = var_name.split('snapshot=')[1].split(',')[0].split(']')[0]
                    if snapshot_part == str(current_snapshot):
                        variable_names.append(var_name)
                        var_indices.append(i)
                else:
                    # Include variables without snapshot dimension (like investment variables)
                    variable_names.append(var_name)
                    var_indices.append(i)

        var_values = self.create_variable_values_mapping(variable_names)

        # Direct mathematical evaluation using only variables from current snapshot
        if var_indices:
            result = np.sum(coeffs_flat[var_indices] *
                        [var_values.get(name, 0) for name in variable_names]) + \
                    self.objective_const
        else:
            # If no variables for this snapshot, just return the constant
            result = self.objective_const

        return result

    def _evaluate_constraint(self, constraint_key):
        """
        Evaluate a single constraint using current network values.

        Parameters:
        -----------
        constraint_key : str
            Key of the specific constraint to evaluate

        Returns:
        --------
        tuple: (bool, float)
            Boolean indicating if constraint is satisfied, and violation amount (0 if satisfied)
        """
        try:
            if constraint_key not in self.constraints:
                return True, 0

            constraint = self.constraints[constraint_key]
            lhs = constraint['lhs']
            rhs = constraint['rhs']
            sign = constraint['sign']
        except Exception as e:
            print(f"Error accessing constraint {constraint_key}: {e}")
            return True, 0  # Default to no violation on error

        # Get variable name mapping for current network
        id_to_name = self.var_id_to_name

        # Evaluate LHS if it's a linear expression
        lhs_value = 0
        if hasattr(lhs, 'vars') and hasattr(lhs, 'coeffs'):
            # Get current variable values
            variable_names = []
            vars_flat = lhs.vars.flatten()  # Flatten to handle any dimensionality
            coeffs_flat = lhs.coeffs.flatten()  # Flatten coefficients as well

            # Create a mask for valid variable indices (not -1)
            valid_indices = vars_flat != -1

            # Filter out -1 indices
            valid_vars = vars_flat[valid_indices]
            valid_coeffs = coeffs_flat[valid_indices]

            # Get variable names only for valid indices
            for var_id in valid_vars:
                if var_id in id_to_name:
                    variable_names.append(id_to_name[var_id])

            # Get variable values
            var_values = self.create_variable_values_mapping(variable_names)

            # Direct mathematical evaluation using only valid variables and coefficients
            try:
                # Convert variable values to numpy array
                var_values_array = np.array([var_values.get(name, 0) for name in variable_names])

                # Calculate LHS value using only valid variables and coefficients
                lhs_value = np.sum(valid_coeffs * var_values_array)

                # Add constant if it exists
                if hasattr(lhs, 'const'):
                    lhs_value += lhs.const.item() if hasattr(lhs.const, 'item') else lhs.const

            except Exception as e:
                print(f"Error in constraint evaluation for {constraint_key}: {e}")
                return True, 0  # Skip this constraint on error
        else:
            # If it's a constant or simple value
            lhs_value = lhs

        # Get RHS value (will be a constant)
        rhs_value = rhs

        # Check constraint satisfaction based on sign
        try:
            # Handle both scalar and array values
            if sign == '==':
                if isinstance(lhs_value, (np.ndarray, list)) or isinstance(rhs_value, (np.ndarray, list)):
                    # Convert to numpy arrays if needed
                    lhs_array = np.asarray(lhs_value)
                    rhs_array = np.asarray(rhs_value)

                    # Handle shape mismatches
                    if lhs_array.shape != rhs_array.shape:
                        if np.isscalar(lhs_array) or len(lhs_array.shape) == 0:
                            lhs_array = np.full_like(rhs_array, lhs_array)
                        elif np.isscalar(rhs_array) or len(rhs_array.shape) == 0:
                            rhs_array = np.full_like(lhs_array, rhs_array)
                        else:
                            # If shapes still don't match, return no violation
                            print(f"Shape mismatch in constraint {constraint_key}: {lhs_array.shape} vs {rhs_array.shape}")
                            return True, 0

                    satisfied = np.all(np.isclose(lhs_array, rhs_array))
                    violation = float(np.sum(np.abs(lhs_array - rhs_array)))
                else:
                    satisfied = np.isclose(lhs_value, rhs_value)
                    violation = float(abs(lhs_value - rhs_value))
            elif sign == '<=':
                if isinstance(lhs_value, (np.ndarray, list)) or isinstance(rhs_value, (np.ndarray, list)):
                    # Convert to numpy arrays if needed
                    lhs_array = np.asarray(lhs_value)
                    rhs_array = np.asarray(rhs_value)

                    # Handle shape mismatches
                    if lhs_array.shape != rhs_array.shape:
                        if np.isscalar(lhs_array) or len(lhs_array.shape) == 0:
                            lhs_array = np.full_like(rhs_array, lhs_array)
                        elif np.isscalar(rhs_array) or len(rhs_array.shape) == 0:
                            rhs_array = np.full_like(lhs_array, rhs_array)
                        else:
                            # If shapes still don't match, return no violation
                            print(f"Shape mismatch in constraint {constraint_key}: {lhs_array.shape} vs {rhs_array.shape}")
                            return True, 0

                    satisfied = np.all(lhs_array <= rhs_array)
                    violation = float(np.sum(np.maximum(0, lhs_array - rhs_array)))
                else:
                    satisfied = lhs_value <= rhs_value
                    violation = float(max(0, lhs_value - rhs_value))
            elif sign == '>=':
                if isinstance(lhs_value, (np.ndarray, list)) or isinstance(rhs_value, (np.ndarray, list)):
                    # Convert to numpy arrays if needed
                    lhs_array = np.asarray(lhs_value)
                    rhs_array = np.asarray(rhs_value)

                    # Handle shape mismatches - don't think I need this
                    if lhs_array.shape != rhs_array.shape:
                        if np.isscalar(lhs_array) or len(lhs_array.shape) == 0:
                            lhs_array = np.full_like(rhs_array, lhs_array)
                        elif np.isscalar(rhs_array) or len(rhs_array.shape) == 0:
                            rhs_array = np.full_like(lhs_array, rhs_array)
                        else:
                            # If shapes still don't match, return no violation
                            print(f"Shape mismatch in constraint {constraint_key}: {lhs_array.shape} vs {rhs_array.shape}")
                            return True, 0

                    satisfied = np.all(lhs_array >= rhs_array)
                    violation = float(np.sum(np.maximum(0, rhs_array - lhs_array)))
                else:
                    satisfied = lhs_value >= rhs_value
                    violation = float(max(0, rhs_value - lhs_value))
            else:
                # Unknown sign
                satisfied = True
                violation = 0

            # Ensure violation is a scalar
            if hasattr(violation, '__len__'):
                violation = float(np.sum(violation))

        except Exception as e:
            print(f"Error comparing constraint values for {constraint_key}: {e}")
            print(f"LHS type: {type(lhs_value)}, RHS type: {type(rhs_value)}")
            print(f"LHS: {lhs_value}, RHS: {rhs_value}")
            satisfied = True
            violation = 0

        return satisfied, violation

    def _evaluate_all_constraints(self):
        """
        Evaluate constraints relevant to the current snapshot.
        Only includes constraints that:
        1. Are for the current snapshot only, or
        2. Link the current snapshot with previous snapshots (but not future snapshots)

        Returns:
        --------
        dict: Information about constraint violations
        """
        results = {
            'all_satisfied': True,
            'violations': {},
            'total_violation': 0.0,
            'violations_by_group': {}  # Track violations by constraint group
        }

        try:
            # Get the current snapshot name
            current_snapshot = self.network.snapshots[self.snapshot_idx]

            # Evaluate each individual constraint that's relevant to the current snapshot
            for constraint_key in self.constraints:
                try:
                    # Check if this constraint is relevant to the current snapshot
                    is_relevant = False

                    # If constraint has no snapshot specification, include it
                    if 'snapshot=' not in constraint_key:
                        is_relevant = True
                    else:
                        # Extract all snapshots mentioned in this constraint
                        constraint_snapshots = []
                        parts = constraint_key.split('snapshot=')
                        for i in range(1, len(parts)):
                            snapshot_val = parts[i].split(',')[0].split(']')[0]
                            constraint_snapshots.append(snapshot_val)

                        # Include if current snapshot is mentioned
                        if str(current_snapshot) in constraint_snapshots:
                            # Check if any future snapshots are mentioned
                            has_future_snapshots = False
                            for snap in constraint_snapshots:
                                try:
                                    # Find the index of this snapshot in the network's snapshots
                                    snap_idx = list(self.network.snapshots).index(snap)
                                    if snap_idx > self.snapshot_idx:
                                        has_future_snapshots = True
                                        break
                                except (ValueError, TypeError):
                                    # If snapshot can't be found or compared, skip this check
                                    pass

                            # Include if no future snapshots are mentioned
                            if not has_future_snapshots:
                                is_relevant = True

                    # Only evaluate if the constraint is relevant
                    if is_relevant:
                        satisfied, violation = self._evaluate_constraint(constraint_key)
                        if not satisfied:
                            results['all_satisfied'] = False
                            results['violations'][constraint_key] = float(violation)  # Ensure it's a scalar
                            results['total_violation'] += float(violation)

                            # Also track violations by constraint group (original name without coordinates)
                            if '[' in constraint_key:
                                group_name = constraint_key.split('[')[0]
                                if group_name not in results['violations_by_group']:
                                    results['violations_by_group'][group_name] = 0.0
                                results['violations_by_group'][group_name] += float(violation)
                except Exception as e:
                    print(f"Error evaluating constraint {constraint_key}: {e}")
                    # Continue with other constraints
        except Exception as e:
            print(f"Error in constraint evaluation: {e}")
            # Return default results

        return results

    def calculate_constrained_reward(self):
        """
        Calculate reward with constraint violation penalties.

        Returns:
        --------
        float: Reward value with constraint penalties
        """
        try:
            # Get base reward from objective function
            base_reward = self._calculate_reward()

            # Evaluate constraints
            constraint_results = self._evaluate_all_constraints()

            # Apply penalty for constraint violations
            # Using a high penalty factor to make violations very noticeable
            # Increased from 100.0 to make violations more obvious

            # Ensure total_violation is a scalar
            total_violation = float(constraint_results['total_violation'])
            penalty = self.penalty_factor * total_violation

            # Final reward is base reward minus penalties
            constrained_reward = base_reward - penalty

            # Ensure reward is a scalar
            if hasattr(constrained_reward, '__len__'):
                constrained_reward = float(constrained_reward)

            return constrained_reward, constraint_results
        except Exception as e:
            print(f"Error calculating constrained reward: {e}")
            # Fall back to unconstrained reward
            return self._calculate_reward()

    def _calculate_reward(self):
        """Calculate reward using stored objective components."""
        # Create a minimal mock expression or use your evaluation directly
        return -1 * self._evaluate_stored_objective()

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action: Array of generator setpoints [gen1_setpoint, gen2_setpoint, ...]

        Returns:
            observation: Network state after action
            reward: Reward for this action
            terminated: Whether episode is finished due to task completion
            truncated: Whether episode is finished due to time limit
            info: Additional information
        """
        scaled_action = self.scale_action(action)

        # Update generator setpoints for non-slack generators only
        for i, gen_name in enumerate(self.non_slack_names):
            self.network.generators_t.p_set.iloc[self.snapshot_idx,
                self.network.generators_t.p_set.columns.get_loc(gen_name)] = scaled_action[i]
        # Run power flow to get new network state
        try:
            # You can choose linear or non-linear power flow
            self.network.lpf()  # Linear power flow
            # print(self.network.generators_t.p.loc['now'])  # Commented out to reduce output
            # self.network.pf()  # Non-linear power flow (alternative)

            power_flow_converged = True
        except Exception as e:
            print(f"Power flow failed: {e}")
            power_flow_converged = False

        # Calculate reward using constrained reward function
        reward, constraint_results = self.calculate_constrained_reward()
        #reward=self._calculate_reward()

        # Increment step counters
        self.current_step += 1
        self.snapshot_idx += 1

        # Handle cycling through snapshots
        if self.snapshot_idx >= self.total_snapshots:
            self.snapshot_idx = 0  # Reset to beginning
            self.reset_network()
            #self._test_hidden_references()
            #can run that instead to check if there are hidden references preventing network cleanup


        # Get new observation
        observation = self._get_observation()

        # Check if episode is done
        episode_done = self._check_done()

        # In gymnasium, we need to separate terminated vs truncated
        terminated = False  # Task completion (not applicable here)
        truncated = episode_done  # Time limit reached

        # Additional info
        info = {
            'generator_setpoints': scaled_action,
            'power_flow_converged': power_flow_converged,
            'generator_names': self.non_slack_names,
            'current_step': self.current_step,
            'snapshot_idx': self.snapshot_idx,
            'constraints_satisfied': constraint_results['all_satisfied'],
            'constraint_violations': constraint_results['violations'],
            'total_violation': constraint_results['total_violation']
        }

        return observation, reward, terminated, truncated, info



    def _check_done(self):
        """
        Check if episode should terminate.

        Episode terminates when we've reached the specified episode length.
        """
        if self.current_step >= self.episode_length:
            return True

        # TO DO: add other cases might want to terminate
        # You might want to terminate on:
        # - Power flow convergence failure
        # - Voltage limit violations
        # - Line overloads

        return False

    def seed(self, seed=None):
        """
        Set the random seed for reproducible experiments.
        """
        np.random.seed(seed)
        return [seed]

    def get_constraint_info(self):
        """
        Get detailed information about the constraints in the model.

        Returns:
        --------
        dict: Detailed information about constraints and their current status
        """
        # Evaluate all constraints
        all_results = self._evaluate_all_constraints()

        # Organize constraint information by groups and individual constraints
        constraint_info = {
            'all_satisfied': all_results['all_satisfied'],
            'total_violation': all_results['total_violation'],
            'groups': {},
            'individual': {}
        }

        # Group constraints by their base name
        for constraint_key in self.constraints:
            # Evaluate this specific constraint
            satisfied, violation = self._evaluate_constraint(constraint_key)

            # Store individual constraint info
            constraint_info['individual'][constraint_key] = {
                'satisfied': satisfied,
                'violation': violation,
                'sign': self.constraints[constraint_key]['sign']
            }

            # Also group by constraint type
            if '[' in constraint_key:
                group_name = constraint_key.split('[')[0]
                if group_name not in constraint_info['groups']:
                    constraint_info['groups'][group_name] = {
                        'count': 0,
                        'satisfied_count': 0,
                        'violated_count': 0,
                        'total_violation': 0.0,
                        'max_violation': 0.0,
                        'sign': self.constraints[constraint_key]['sign']
                    }

                group_info = constraint_info['groups'][group_name]
                group_info['count'] += 1

                if satisfied:
                    group_info['satisfied_count'] += 1
                else:
                    group_info['violated_count'] += 1
                    group_info['total_violation'] += violation
                    group_info['max_violation'] = max(group_info['max_violation'], violation)
            else:
                # Handle standalone constraints
                if 'standalone' not in constraint_info['groups']:
                    constraint_info['groups']['standalone'] = {
                        'count': 0,
                        'satisfied_count': 0,
                        'violated_count': 0,
                        'total_violation': 0.0,
                        'max_violation': 0.0
                    }

                group_info = constraint_info['groups']['standalone']
                group_info['count'] += 1

                if satisfied:
                    group_info['satisfied_count'] += 1
                else:
                    group_info['violated_count'] += 1
                    group_info['total_violation'] += violation
                    group_info['max_violation'] = max(group_info['max_violation'], violation)

        return constraint_info

    def render(self, mode='human', info=None):
        """
        Render the environment state.

        Parameters:
        -----------
        mode : str
            Rendering mode (only 'human' supported)
        info : dict, optional
            Information dictionary from step() method containing constraint data
        """
        print("=== Current Network State ===")
        print(f"Episode step: {self.current_step}/{self.episode_length}")
        print(f"Snapshot index: {self.snapshot_idx}/{self.total_snapshots}")
        print(f"Current snapshot: {self.network.snapshots[self.snapshot_idx]}")
        print(f"Generator setpoints: {self.network.generators_t.p_set.iloc[self.snapshot_idx].values}")
        print(f"Load values: {self.network.loads_t.p_set.iloc[self.snapshot_idx].values}")

        all_satisfied = info['constraints_satisfied']
        total_violation = info['total_violation']
        violations = info['constraint_violations']


        print(f"All constraints satisfied: {all_satisfied}")
        print(f"Total constraint violation: {total_violation:.4f}")

        # Show violated constraints if any
        if not all_satisfied and violations:
            print("\n=== Constraint Violations ===")
            for constraint_name, violation in violations.items():
                print(f"  {constraint_name}: {violation:.4f}")

Env2Gen1LoadConstr(network_file="/Users/antoniagrindrod/Documents/pypsa-earth_project/pypsa-earth-RL/networks/elec_s_5_ec_lcopt_3h.nc")