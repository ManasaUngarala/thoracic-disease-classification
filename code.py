# Multi-Label Classification of Thoracic Diseases using DenseNet-121
# Dataset: NIH ChestX-ray14
# Author: Manasa Ungarala

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# === Step 1: Define transformations for image preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Step 2: Load dataset (mock placeholder) ===
# NOTE: In real use, replace with the actual ChestX-ray14 dataset path
train_dataset = ImageFolder(root='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# === Step 3: Load pretrained DenseNet-121 and modify for multi-label output ===
model = models.densenet121(pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 14)  # 14 thoracic disease classes

# === Step 4: Set loss function and optimizer ===
criterion = nn.BCEWithLogitsLoss()  # Best for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Step 5: Train for 1 epoch (demonstration purpose only) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).type(torch.float32)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

print("✅ Training complete for demonstration.")
