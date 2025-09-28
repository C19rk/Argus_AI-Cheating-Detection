import os

# Path to your dataset labels
label_dirs = [
    "data/train/labels",
    "data/valid/labels"
]

expected_numbers = 56  # 1 class + 4 box + 51 keypoints (for 17 keypoints * 3)

for label_dir in label_dirs:
    if not os.path.exists(label_dir):
        print(f"❌ Missing folder: {label_dir}")
        continue

    print(f"\n🔍 Checking labels in: {label_dir}")
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        path = os.path.join(label_dir, file)
        with open(path, "r") as f:
            for i, line in enumerate(f.readlines(), start=1):
                nums = line.strip().split()
                if len(nums) != expected_numbers:
                    print(f"❌ {file} (line {i}) has {len(nums)} numbers, expected {expected_numbers}")
