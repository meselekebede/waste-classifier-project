# predict_waste.py

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np


CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def build_resnet_model(num_classes=len(CLASS_NAMES)):
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(model_path='best_resnet_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_resnet_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

def preprocess_image(image):
    return TRANSFORMS(image).unsqueeze(0)

def predict_with_confidence(model, image_tensor):
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    predicted_idx = np.argmax(probs)
    prediction = CLASS_NAMES[predicted_idx]
    confidence_dict = {cls.capitalize(): float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}
    return prediction, confidence_dict