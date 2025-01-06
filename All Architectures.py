# Import necessary libraries
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b0, mobilenet_v2, shufflenet_v2_x1_0
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from collections import Counter

# Define model training hyperparameters
BATCH_SIZE = 32
EPOCHS = 25
IMAGE_SIZE = 48
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #If possible, use CUDA

def plotConfusionMatrix(cm, dataset_name, normalize=False, class_labels=None):
    """
    Plots a confusion matrix using seaborn's heatmap
    Args:
        cm: confusion matrix array
        dataset_name: name of the dataset for the title
        normalize: whether to normalize the confusion matrix
        class_labels: labels for the classes
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    if class_labels is None:
        class_labels = np.arange(cm.shape[0])
    
    plt.figure(figsize=(12, 8)) 
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='YlGnBu', 
                cbar=True, annot_kws={"size": 10}, linewidths=0.5, linecolor='gray',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix for {dataset_name}', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.show()

data_dir = "PATH TO DATASET GOES HERE" #Specify the dataset path here

# Define image transformations for training and validation
# Convert images to 3-channel grayscale and resize them
train_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)

# Load training and validation datasets
train_dataset = datasets.ImageFolder(
    root=f"{data_dir}/train", transform=train_transform
)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=val_transform)

# Calculate class weights to handle imbalanced dataset
class_counts = Counter(train_dataset.targets)
class_weights = [1.0 / class_counts[i] for i in range(len(train_dataset.classes))]
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# Create weighted sampler to balance the training data
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# Get class names and create data loaders
class_names = train_dataset.classes
dataloaders = {
    "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler),
    "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
}

def get_model(architecture="efficientnet", num_classes=7):
    """
    Creates and returns a neural network model based on the specified architecture
    Args:
        architecture: type of model to use ('efficientnet', 'mobilenetv2', or 'shufflenet')
        num_classes: number of output classes
    Returns:
        initialized model
    """
    if architecture == "efficientnet":
        model = efficientnet_b0(pretrained=True)
        # Modify first conv layer to accept grayscale images
        model.features[0][0] = nn.Conv2d(
            3, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        # Modify classifier for our number of classes
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(model.classifier[1].in_features, num_classes)
        )

        # Enable fine-tuning of all layers
        for param in model.features.parameters():
            param.requires_grad = True
    elif architecture == "mobilenetv2":
        model = mobilenet_v2(pretrained=True)
        # Modify first conv layer
        model.features[0][0] = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.features.add_module("global_pool", nn.AdaptiveAvgPool2d((1, 1)))
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(model.last_channel, num_classes)
        )

        for param in model.features.parameters():
            param.requires_grad = True
    elif architecture == "shufflenet":
        model = shufflenet_v2_x1_0(pretrained=True)
        # Modify first conv layer
        model.conv1[0] = nn.Conv2d(
            3, 24, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.features.parameters():
            param.requires_grad = True
    else:
        raise ValueError(
            "Invalid architecture. Choose 'efficientnet', 'mobilenetv2', 'shufflenet'."
        )

    model.to(DEVICE)
    return model

def train_and_validate(
    model, criterion, optimizer, dataloaders, architecture, num_epochs=EPOCHS
):
    """
    Trains and validates the model
    Args:
        model: neural network model
        criterion: loss function
        optimizer: optimization algorithm
        dataloaders: dictionary containing train and validation dataloaders
        architecture: model architecture name
        num_epochs: number of training epochs
    """
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate training statistics
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels).item()
            train_total += labels.size(0)

        # Calculate training metrics
        train_accuracy = train_correct / train_total
        train_loss /= train_total
        print(f"Train Loss: {train_loss:.4f} Accuracy: {train_accuracy * 100:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []

        # No gradient calculation during validation
        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Calculate validation statistics
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Calculate validation metrics
        val_accuracy = val_correct / val_total
        val_loss /= val_total
        print(f"Validation Loss: {val_loss:.4f} Accuracy: {val_accuracy * 100:.2f}%")

        end_time = time.time()
        exec_time = end_time - start_time
        print(f"Epoch execution time: {exec_time:.2f} seconds")

    # Generate and save final classification report and confusion matrix
    print("Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    print("Confusion Matrix:")
    confusion = confusion_matrix(all_labels, all_preds)
    print(confusion)

    # Save results to file
    with open(f"./PATH WHERE YOU WANT TO SAVE IT GOES HERE/{architecture}.txt", "a") as f: #Specify where to save results
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(confusion, separator=", ") + "\n\n")

# Main execution loop
while (True): #infinite loop
    for architecture in ["efficientnet", "mobilenetv2", "shufflenet"]:
        print(f"Training and validating model: {architecture}")
        # Initialize model, loss function, and optimizer
        model = get_model(architecture, num_classes=len(class_names))
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
        ) 

        # Train and validate the model
        train_and_validate(
            model,
            criterion,
            optimizer,
            dataloaders,
            architecture,
            num_epochs=EPOCHS,
        )