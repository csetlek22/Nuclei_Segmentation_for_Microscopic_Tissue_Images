import os
import numpy as np

# Path to the pairs file we generated
pairs_file = "image_mask_pairs.txt"

# Read all (image, mask) lines
with open(pairs_file, "r") as f:
    all_lines = [line.strip() for line in f.readlines() if line.strip()]

# Replace backslashes with forward slashes in all lines
all_lines = [line.replace('\\', '/') for line in all_lines]

# Shuffle all lines
np.random.seed(42)
np.random.shuffle(all_lines)

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

total = len(all_lines)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_lines = all_lines[:train_end]
val_lines = all_lines[train_end:val_end]
test_lines = all_lines[val_end:]

# Create output directory
split_dir = "folds_collab"
os.makedirs(split_dir, exist_ok=True)

# Save splits to txt files
with open(os.path.join(split_dir, "train.txt"), "w") as f:
    f.write("\n".join(train_lines))

with open(os.path.join(split_dir, "val.txt"), "w") as f:
    f.write("\n".join(val_lines))

with open(os.path.join(split_dir, "test.txt"), "w") as f:
    f.write("\n".join(test_lines))

print(f"✅ Dataset split completed:")
print(f" - Train samples: {len(train_lines)}")
print(f" - Validation samples: {len(val_lines)}")
print(f" - Test samples: {len(test_lines)}")
