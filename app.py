from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import torchvision.transforms as transforms
from io import BytesIO
import base64
import numpy as np

from model import UNet


app = FastAPI(title="Oil Spill Detection API")


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


# Same preprocessing used during testing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


@app.get("/")
def home():
    return {
        "message": "Oil Spill Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_data = await file.read()

    image = Image.open(
        BytesIO(image_data)
    ).convert("RGB")

    original_size = image.size

    # Preprocess
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output)
        prediction = (probability > 0.5).float()

    # Convert prediction to numpy
    mask = prediction.squeeze().cpu().numpy()

    # Calculate spill percentage
    total_pixels = mask.size
    spill_pixels = np.sum(mask == 1)

    spill_percentage = (
        spill_pixels / total_pixels
    ) * 100

    # Determine whether oil spill exists
    oil_spill_detected = bool(spill_pixels > 0)

    # Convert mask to image
    mask_image = Image.fromarray(
        (mask * 255).astype("uint8")
    )

    # Resize mask back to original image size
    mask_image = mask_image.resize(
        original_size,
        Image.NEAREST
    )

    # Convert mask to Base64
    buffer = BytesIO()

    mask_image.save(
        buffer,
        format="PNG"
    )

    mask_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


    return JSONResponse({
        "success": True,
        "filename": file.filename,
        "original_size": original_size,
        "oil_spill_detected": oil_spill_detected,
        "spill_percentage": round(float(spill_percentage), 2),
        "mask": mask_base64
    })