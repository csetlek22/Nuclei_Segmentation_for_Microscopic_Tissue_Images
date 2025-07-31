import os
from sklearn.model_selection import KFold

# Path to the pairs file we just generated
pairs_file = "image_mask_pairs.txt"

# Read all (image, mask) lines
with open(pairs_file, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# Create output directory for splits
split_dir = "folds"
os.makedirs(split_dir, exist_ok=True)

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(lines)):
    train_lines = [lines[i] for i in train_idx]
    val_lines = [lines[i] for i in val_idx]

    # Save to text files
    with open(os.path.join(split_dir, f"train_fold{fold_idx}.txt"), "w") as f:
        f.write("\n".join(train_lines))

    with open(os.path.join(split_dir, f"val_fold{fold_idx}.txt"), "w") as f:
        f.write("\n".join(val_lines))

    print(f"✅ Saved fold {fold_idx}: {len(train_lines)} train, {len(val_lines)} val")
