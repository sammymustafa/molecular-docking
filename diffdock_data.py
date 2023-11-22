from setup import *

DIFFDOCK_DATA = "/content/diffdock/diffdock/"
GT_PATH = "6o5g_B_LMJ.sdf"

ground_truth = os.path.join(DIFFDOCK_DATA, GT_PATH)
predictions = []
for file_name in os.listdir(DIFFDOCK_DATA):
  if "rank" not in file_name:
    continue
  else:
    meta = file_name.split("_")
    group = meta[0]
    rank = int(meta[2][-1])
    confidence = float(meta[3].split("confidence")[1][:-4])
    predictions.append({
        "path": os.path.join(DIFFDOCK_DATA, file_name),
        "group": group,
        "rank": rank,
        "confidence": confidence
    })

print("Ground truth path:\n")
print(ground_truth)

print("\nPredictions:")
print(pd.DataFrame(predictions))