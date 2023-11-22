from setup import *

# Initialize a list to hold RMSD values and corresponding file paths
rmsd_values = []

# Iterate over each prediction file
for prediction in predictions:
    # Load the predicted molecule
    predicted_mol = load_coordinates_sdf(prediction["path"])

    # Compute the RMSD with the ground truth
    rmsd = compute_rmsd(ground_truth_mol, predicted_mol)

    # Store the RMSD and file path
    rmsd_values.append((rmsd, prediction["path"]))

# Sort the list by RMSD values
rmsd_values.sort(key=lambda x: x[0])

# Output the best and worst RMSD values and their corresponding file paths
best_rmsd, best_prediction_path = rmsd_values[0]
worst_rmsd, worst_prediction_path = rmsd_values[-1]

print(f"Best RMSD: {best_rmsd}, File: {best_prediction_path}")
print(f"Worst RMSD: {worst_rmsd}, File: {worst_prediction_path}")

# Set the best file and download it by running this cell
best_prediction = "5_iter_rank1_confidence-3.24.sdf"
worst_prediction = "5_iter_rank5_confidence-4.81.sdf"

# Download
from google.colab import files

DIFFDOCK_DATA = "/content/diffdock/diffdock/"
files.download(os.path.join(DIFFDOCK_DATA, best_prediction))
files.download(os.path.join(DIFFDOCK_DATA, worst_prediction))
files.download(os.path.join(DIFFDOCK_DATA, "6O5G.pdb"))