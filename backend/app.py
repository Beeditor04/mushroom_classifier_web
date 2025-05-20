from fastapi import FastAPI
from contextlib import asynccontextmanager
import wandb
import torch
import os
from utils.helper import get_model
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and store in app.state
    api = wandb.Api()
    id = "convnext-model"
    version = "best"
    artifact_model = api.artifact(
        f"hutech_mushroom/{id}:{version}",
        type="model",
    )
    artifact_model_dir = artifact_model.download()
    model = get_model("convnext", num_classes=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_name = os.listdir(artifact_model_dir)
    model.load_state_dict(torch.load(os.path.join(artifact_model_dir, model_name[0]), map_location=device))
    
    #--- save loaded stuff to app
    app.state.model = model
    app.state.device = device
    print("Model load completed.")
    yield  
    print("Sucessfully kill yourself.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:5173"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Example endpoint
from fastapi import UploadFile, File
from PIL import Image
from torchvision.transforms import transforms
import io

@app.post("/predict")
# you can run test on localhost:8000/docs
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    model = app.state.model
    device = app.state.device

    # load transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    #construct img tensor
    img_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    THRESHOLD = 0.7
    with torch.no_grad(): #predict w no grad
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1)
        # Check if the probability is above the threshold
        if probabilities[0][predicted_idx].item() < THRESHOLD:
            return {"predicted_class": "Can not find any mushroom."}
        print("Output:", output)
        print("Conf:", probabilities)
        print(torch.max(output, 1))
        print(predicted_idx)

    classes = {
        0: "nấm mỡ",
        1: "nấm bào ngư",
        2: "nấm đùi gà",
        3: "nấm linh chi trắng",
    }
    return {"predicted_class": classes[predicted_idx.item()]}