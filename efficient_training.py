import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
from model import EfficientNetClassifier  # Imported from model.py

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Datasets & Loaders
train_data = datasets.ImageFolder("/content/drive/MyDrive/NEW_MODEL_01/Output/train", transform=train_transform)
val_data = datasets.ImageFolder("/content/drive/MyDrive/NEW_MODEL_01/Output/val", transform=val_transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=2)

# Class weights
train_labels = np.array(train_data.targets)
weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
weights = torch.tensor(weights, dtype=torch.float).to(device)

# Model, Loss, Optimizer
model = EfficientNetClassifier(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training
best_val_loss = float('inf')
patience, counter = 5, 0
EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = mse ** 0.5

    print(f"\n📊 Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f} | Val Loss={val_loss/len(val_loader):.4f}")
    print(f"✅ Accuracy={acc:.4f} | Kappa={kappa:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}\n")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "/content/drive/MyDrive/NEW_MODEL_01/efficientnet_b0_best.pth")
    else:
        counter += 1
        if counter >= patience:
            print("⛔ Early stopping triggered.")
            break
