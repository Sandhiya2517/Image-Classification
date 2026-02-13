# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Import the necessary libraries such as NumPy, Matplotlib, and PyTorch.

### STEP 2:
Load and preprocess the dataset:

Resize images to a fixed size (128×128).
Normalize pixel values to a range between 0 and 1.
Convert labels into numerical format if necessary.
### STEP 3:
Define the CNN Architecture, which includes:

Input Layer: Shape (8,128,128)
Convolutional Layer 1: 8 filters, kernel size (16×16), ReLU activation
Max-Pooling Layer 1: Pool size (2×2)
Convolutional Layer 2: 24 filters, kernel size (8×8), ReLU activation
Max-Pooling Layer 2: Pool size (2×2)
Fully Connected (Dense) Layer:
First Dense Layer with 256 neurons
Second Dense Layer with 128 neurons
Output Layer for classification
### STEP 4:
Define the loss function (e.g., Cross-Entropy Loss for classification) and optimizer (e.g., Adam or SGD).

### STEP 5:
Train the model by passing training data through the network, calculating the loss, and updating the weights using backpropagation.

### STEP 6:
Evaluate the trained model on the test dataset using accuracy, confusion matrix, and other performance metrics.

### STEP 7:
Make predictions on new images and analyze the results.


## PROGRAM

### Name: SANDHIYA M
### Register Number: 212224220086

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nName: SANDHIYA M")
print("Register Number: 212224220086")
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

accuracy = accuracy_score(all_labels, all_preds)
print("accuracy                           {:.2f}     {}".format(accuracy, len(all_labels)))

index = 0
image, label = test_dataset[index]

with torch.no_grad():
    output = model(image.unsqueeze(0).to(device))
    _, pred = torch.max(output, 1)

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Actual: {class_names[label]}\nPredicted: {class_names[pred.item()]}")
plt.axis('off')
plt.show()

```

## OUTPUT
### Training Loss per Epoch

<img width="389" height="95" alt="Screenshot 2026-02-13 155115" src="https://github.com/user-attachments/assets/911d5a7e-9563-429c-8e8c-287b686e7310" />

### Confusion Matrix

<img width="815" height="763" alt="Screenshot 2026-02-13 155154" src="https://github.com/user-attachments/assets/a3a736f3-e4f1-48e5-8e1c-f2c8e8f4fe91" />

### Classification Report

<img width="553" height="407" alt="Screenshot 2026-02-13 155209" src="https://github.com/user-attachments/assets/8d187615-348a-43fa-8068-8782f4cb3fe6" />

### New Sample Data Prediction

<img width="394" height="436" alt="Screenshot 2026-02-13 155218" src="https://github.com/user-attachments/assets/04b08ce0-0598-49bd-98dc-69da2c6fa19a" />


## RESULT

The Convolutional Neural Network (CNN) was successfully implemented for image classification. The model was trained on the dataset, and its performance was evaluated using accuracy metrics, confusion matrix, and classification report. Predictions on new sample images were verified, confirming the model's effectiveness in classifying images.
