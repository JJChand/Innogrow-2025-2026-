import torch

# Fix PyTorch 2.8 compatibility issue by temporarily overriding torch.load
# This is needed because ultralytics models were saved with older PyTorch versions
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    # Force weights_only=False for ultralytics model loading
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)

# Apply the patch
torch.load = patched_torch_load

# Now import ultralytics after patching
from ultralytics import YOLO

# Load your trained model
model_path = '(example)runs/detect/train/weights/best.pt'
print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# Restore original torch.load (good practice)
torch.load = original_torch_load

# Export to ONNX
print("Exporting model to ONNX format...")
model.export(format='onnx')
print("Export completed successfully!")

