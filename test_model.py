import torch
from model import UNet

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Create the exact same architecture used during training
model = UNet(n_channels=3, n_classes=1)

# Load trained weights
model.load_state_dict(
    torch.load(
        "oil_spill_unet.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("Model loaded successfully!")
print("Model is ready for inference.")