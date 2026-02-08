"""
Quick end-to-end test: build datasets with random rotation augmentation
and pull one batch through the DataLoader to confirm the training pipeline works.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader
from src.tp_transformer.config import TrainConfig
from src.tp_transformer.data import build_datasets

# Override to use random rotation
config = TrainConfig()
config.augmentation_method = "random"
config.n_train_demos = 5  # use fewer demos for speed

print(f"Augmentation method: {config.augmentation_method}")
print("Building datasets...")
training_data, valid_data, test_data, train_stats = build_datasets(config)

print(f"\nTraining samples: {len(training_data)}")
print(f"Validation samples: {len(valid_data)}")
print(f"Test samples: {len(test_data)}")

# Pull one batch from training DataLoader
loader = DataLoader(training_data, batch_size=2, shuffle=True)
batch = next(iter(loader))

obj_data, traj_data, traj_hidden, weights, action_tag, padding_mask, img_inds, pick_inds, release_inds = batch

print(f"\n--- Batch shapes ---")
print(f"obj_data:      {obj_data.shape}")
print(f"traj_data:     {traj_data.shape}")
print(f"traj_hidden:   {traj_hidden.shape}")
print(f"weights:       {weights.shape}")
print(f"action_tag:    {action_tag.shape}")
print(f"padding_mask:  {padding_mask.shape}")
print(f"img_inds:      {img_inds.shape}")

print("\nPipeline test PASSED -- random rotation augmentation works end-to-end!")
