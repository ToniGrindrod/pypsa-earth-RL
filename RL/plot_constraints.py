#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define constraint names based on the requirements
constraint_labels = {
    "0": "p_min_ZA0 0 coal",
    "2": "p_max_ZA0 0 coal",
    "4": "soc_min_ZA0 0 PHS"
}

# Define the keys
keys = {
    "EL-13": "EnvDispatchReplacement",
    "EL-14": "EnvDispatchConstr"
}

# Create directory for output plots if it doesn't exist
os.makedirs("RL/constraint_plots", exist_ok=True)

# Process each constraint number
for constraint_num in ["0", "2", "4"]:
    plt.figure(figsize=(10, 6))
    
    # Process each key (EL-13 and EL-14)
    for key in ["EL-13", "EL-14"]:
        # Read the CSV file
        file_path = f"RL/constraints/{key}__constraint_{constraint_num}.csv"
        
        # Read data
        df = pd.read_csv(file_path, header=None, names=["step", "timestamp", "value"])
        
        # Convert step to integer for plotting
        df["step"] = df["step"].astype(int)
        
        # Plot the data
        plt.plot(df["step"], df["value"], marker='o', label=f"{keys[key]}")
    
    # Set plot properties
    plt.title(f"{constraint_labels[constraint_num]}")
    plt.xlabel("Step")
    plt.ylabel("Total Violation")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(np.arange(0, max(df["step"])+1, 1))  # Only integer steps
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"RL/constraint_plots/constraint_{constraint_num}.png")
    plt.close()

print("Plots generated in RL/constraint_plots/ directory")
