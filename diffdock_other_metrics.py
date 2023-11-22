from setup import *

# Initialize dictionaries to hold values for each metric
mean_rmsd_per_group = {'5': [], '10': [], '20': []}
best_rmsd_per_group = {'5': float('inf'), '10': float('inf'), '20': float('inf')}
highest_confidence_rmsd_per_group = {'5': (-float('inf'), float('inf')), '10': (-float('inf'), float('inf')), '20': (-float('inf'), float('inf'))}

# Process each prediction
for prediction in predictions:
    # Load the predicted molecule
    predicted_mol = load_coordinates_sdf(prediction["path"])

    # Compute the RMSD with the ground truth
    rmsd = compute_rmsd(ground_truth_mol, predicted_mol)

    # Update the best and mean RMSD for the group
    group = prediction["group"]
    mean_rmsd_per_group[group].append(rmsd)
    if rmsd < best_rmsd_per_group[group]:
        best_rmsd_per_group[group] = rmsd

    # Update the highest confidence RMSD for the group
    confidence = prediction["confidence"]
    if confidence > highest_confidence_rmsd_per_group[group][0]:  # This is correct if less negative means higher confidence
        highest_confidence_rmsd_per_group[group] = (confidence, rmsd)

# Calculate mean RMSDs
for group in mean_rmsd_per_group:
    mean_rmsd_per_group[group] = sum(mean_rmsd_per_group[group]) / len(mean_rmsd_per_group[group])

# Extract RMSD from the highest confidence tuples
for group in highest_confidence_rmsd_per_group:
    highest_confidence_rmsd_per_group[group] = highest_confidence_rmsd_per_group[group][1]

# Print the results
print("Performance metrics per group:")
print("Mean RMSD:", mean_rmsd_per_group)
print("Best RMSD:", best_rmsd_per_group)
print("Highest Confidence RMSD:", highest_confidence_rmsd_per_group)