import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Note: Hybrid CNN + SVM architectures may not work well. Additionally, SVM drastically increases the computational cost; I might use the CNN variations only.

# Define hyperparameters
BATCH_SIZE = 64
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset preparation
data_dir = "path_to_dataset"
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
}

# MobileNetV2 model for feature extraction
def get_feature_extractor():
    model = mobilenet_v2(pretrained=True)
    model.classifier = nn.Identity()  # Remove the classification layer
    model.eval()
    model.to(DEVICE)
    return model

# Extract features using MobileNetV2
def extract_features(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)

# Initialize MobileNetV2 model
feature_extractor = get_feature_extractor()

# Extract features from train and validation sets
print("Extracting train features...")
train_features, train_labels = extract_features(feature_extractor, dataloaders['train'])
print("Extracting validation features...")
val_features, val_labels = extract_features(feature_extractor, dataloaders['val'])

# Standardize features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

# Train SVM classifier
print("Training SVM...")
svm_clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm_clf.fit(train_features, train_labels)

# Validate SVM classifier
print("Validating SVM...")
val_predictions = svm_clf.predict(val_features)
accuracy = accuracy_score(val_labels, val_predictions)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(val_labels, val_predictions, target_names=train_dataset.classes))

# Save the trained SVM model
import joblib
joblib.dump(svm_clf, "svm_model_mobilenet.pkl")
print("SVM model saved as svm_model_mobilenet.pkl")