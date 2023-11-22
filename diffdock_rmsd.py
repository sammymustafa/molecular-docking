from setup import *

# HELPERS
def load_coordinates_sdf(path):
    """Load molecule from sdf."""
    suppl = Chem.SDMolSupplier(path)
    mol = next(suppl)
    return mol

def compute_rmsd(mol1, mol2):
    """Compute the RMSD between mol1 and mol2."""
    return Chem.rdMolAlign.CalcRMS(mol1, mol2)

# Load the ground truth molecule
ground_truth_mol = load_coordinates_sdf(ground_truth)

# Initialize a dictionary to hold the best RMSD for each group
best_rmsd_per_group = {'5': float('inf'), '10': float('inf'), '20': float('inf')}

# Process each prediction
for prediction in predictions:
    # Load the predicted molecule
    predicted_mol = load_coordinates_sdf(prediction["path"])

    # Compute the RMSD with the ground truth
    rmsd = compute_rmsd(ground_truth_mol, predicted_mol)

    # Update the best RMSD for the group if this RMSD is lower
    group = prediction["group"]
    if rmsd < best_rmsd_per_group[group]:
        best_rmsd_per_group[group] = rmsd

# Print the best RMSD for each group
print("Best RMSD per group:")
for group, rmsd in best_rmsd_per_group.items():
    print(f"Group {group}: {rmsd}")