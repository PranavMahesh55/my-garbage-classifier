import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

app = FastAPI()

CLASS_NAMES = ['cardboard','glass','metal','paper','plastic','trash']

model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("model.pt"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 192)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(400, "File is not an image.")
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
        label = CLASS_NAMES[pred.item()]
        probs = torch.nn.functional.softmax(outputs, dim=1)[0, pred].item()

    return JSONResponse({"label": label, "confidence": probs})

@app.get("/")
def healthcheck():
    return {"status": "ok"}