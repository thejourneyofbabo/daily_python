import os
import pandas as pd
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 1. Loading Built-in Datasets
print("=== Loading Built-in Datasets ===")

# Download FashionMNIST dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 2. Data Visualization
print("\n=== Data Visualization ===")

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    
plt.show()

# 3. Creating Custom Dataset
print("\n=== Creating Custom Dataset ===")

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label

# 4. Preparing Data with DataLoader
print("\n=== Preparing Data with DataLoader ===")

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 5. Iterating Through Data and Visualization
print("\n=== Iterating Through Data and Visualization ===")

# Display features and labels for the first batch
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# Visualize a sample image
img = train_features[0].squeeze()
label = train_labels[0]
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap="gray")
plt.title(f"Label: {labels_map[label.item()]}")
plt.axis("off")
plt.show()

print(f"Label: {labels_map[label.item()]}")

# 6. Transforms Example
print("\n=== Transforms Example ===")

from torchvision import transforms

# Define various transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalization
])

# Dataset with transforms applied
transformed_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# Compare before and after transformation
original_img, label = training_data[0]
transformed_img, _ = transformed_dataset[0]

print(f"Original image range: [{original_img.min():.3f}, {original_img.max():.3f}]")
print(f"Transformed image range: [{transformed_img.min():.3f}, {transformed_img.max():.3f}]")

# 7. Data Loading Performance Optimization Example
print("\n=== Data Loading Performance Optimization ===")

# DataLoader with multiprocessing
fast_dataloader = DataLoader(
    training_data, 
    batch_size=64, 
    shuffle=True,
    num_workers=2,  # Multiprocessing
    pin_memory=True  # Memory optimization for GPU usage
)

# Measure data loading time
import time

start_time = time.time()
for batch_idx, (data, target) in enumerate(fast_dataloader):
    if batch_idx >= 10:  # Measure only 10 batches
        break
    pass

end_time = time.time()
print(f"Time to load 10 batches: {end_time - start_time:.2f} seconds")

print("\n=== Tutorial Complete ===")
print("Now you know how to effectively load and process data in PyTorch!")
