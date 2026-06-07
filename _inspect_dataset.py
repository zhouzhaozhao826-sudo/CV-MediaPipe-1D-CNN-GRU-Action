import numpy as np

data = np.load(
    "datasets/round3/exports_round3/_merged/round3_merged_4class/round3_merged_4class_dataset.npz",
    allow_pickle=True,
)

print("=== 文件中的数组键 ===")
print(list(data.keys()))
print()

X = data["X"]
y = data["y"]
feature_names = data["feature_names"]
seq_len = data["sequence_length"]

print(f"X shape: {X.shape}")
print(f"  -> {X.shape[0]} samples, {X.shape[1]} frames, {X.shape[2]} features")
print(f"y shape: {y.shape}")
print(f"sequence_length: {seq_len}")
print()

print("=== 标签分布 ===")
label_map = {
    0: "0-standard",
    1: "1-depth_insufficient",
    2: "2-knee_valgus",
    3: "3-torso_lean",
}
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  label {u} ({label_map.get(u, '?')}): {c} samples")
print()

print("=== 15 features ===")
for i, name in enumerate(feature_names):
    print(f"  [{i:2d}] {name}")
print()

print("=== X[0] first 5 frames x first 8 features ===")
print(X[0][:5, :8])
print()
print(f"X[0] label: {y[0]}")
print()

print("=== X[50] first 5 frames x first 8 features ===")
print(X[50][:5, :8])
print(f"X[50] label: {y[50]}")
print()

print("=== X[100] first 5 frames x first 8 features ===")
print(X[100][:5, :8])
print(f"X[100] label: {y[100]}")
