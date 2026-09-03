import torch
from PIL import Image
import torchvision.transforms as transforms

from model import UNet


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model
model = UNet(n_channels=3, n_classes=1)

model.load_state_dict(
    torch.load(
        "oil_spill_unet.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# Load test image
image = Image.open("test_image.png").convert("RGB")

original_size = image.size

input_tensor = transform(image).unsqueeze(0).to(device)


# Prediction
with torch.no_grad():
    output = model(input_tensor)
    probability = torch.sigmoid(output)
    prediction = (probability > 0.5).float()


# Remove batch/channel dimensions
mask = prediction.squeeze().cpu().numpy()


# Convert to PNG
mask_image = Image.fromarray((mask * 255).astype("uint8"))

mask_image.save("predicted_mask_1.png")


print("Inference successful!")
print("Original image size:", original_size)
print("Prediction size:", mask_image.size)
print("Saved: predicted_mask_1.png")