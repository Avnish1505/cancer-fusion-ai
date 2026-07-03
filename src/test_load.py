import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 7)  # 7 classes
model = model.to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

print("Model loaded successfully!")
print(model.fc)