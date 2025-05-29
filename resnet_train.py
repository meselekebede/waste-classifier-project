import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import resnet34, ResNet34_Weights
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import time

# ==== CONFIG ====
IMAGE_SIZE = 224  # Default input size for ResNet-34
SRC_DIR = "dataset"  # Path to dataset directory
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42

# Ensure reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==== DATASET PREPARATION ====
def prepare_resnet_dataloaders(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, src_dir=SRC_DIR):
    """
    Prepares data loaders for training, validation, and testing.
    Applies transformations to resize, normalize, and convert images to tensors.
    Splits the dataset into 80% training, 10% validation, and 10% testing.
    """
    if not os.path.exists(src_dir) or len(os.listdir(src_dir)) == 0:
        raise ValueError(f"Dataset directory {src_dir} is missing or empty.")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=src_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset.classes


# ==== MODEL ====
def build_resnet_model(num_classes, device=None):
    """
    Builds a ResNet-34 model with a custom output layer for the specified number of classes.
    Freezes all base layers and replaces the final classification layer.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classification layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model, device


# ==== EARLY STOPPING ====
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='best_resnet_model.pth', verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ==== TRAINING FUNCTION ====
def train_resnet(model, device, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, early_stopping_patience=EARLY_STOPPING_PATIENCE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, cooldown=2)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path='best_resnet_model.pth')

    train_losses, train_accuracies = [], []
    val_accuracies, val_f1_scores, val_precisions, val_recalls = [], [], [], []
    learning_rates = []

    scaler = GradScaler()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Train Acc: {100 * epoch_acc:.2f}%")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_precision = precision_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')

        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

        print(f"Validation Accuracy: {100 * val_acc:.2f}%, F1 Score: {val_f1:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        scheduler.step(epoch_loss)
        early_stopping(epoch_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load('best_resnet_model.pth'))
    return train_losses, train_accuracies, val_accuracies, val_f1_scores, val_precisions, val_recalls, learning_rates


# ==== TEST SET EVALUATION ====
def evaluate_test_set(model, device, test_loader):
    """
    Evaluates the model on the test set and prints accuracy, F1 score, precision, and recall.
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = correct / total
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')

    print(f"\nTest Set Evaluation:")
    print(f"Accuracy: {100 * test_acc:.2f}%, F1 Score: {test_f1:.4f}, "
          f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")


# ==== MAIN EXECUTION ====
if __name__ == '__main__':
    start_time = time.time()

    # Prepare data loaders
    train_loader, val_loader, test_loader, class_names = prepare_resnet_dataloaders()

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_resnet_model(num_classes=len(class_names), device=device)

    # Train model
    train_losses, train_accuracies, val_accuracies, val_f1_scores, val_precisions, val_recalls, learning_rates = \
        train_resnet(model, device, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)

    # Evaluate on test set
    evaluate_test_set(model, device, test_loader)

    # Save final model
    torch.save(model.state_dict(), "final_resnet_model.pth")

    # Training time
    end_time = time.time()
    training_time = round((end_time - start_time) / 60, 2)  # in minutes
    print(f"\nTotal Training Time: {training_time} minutes")

    # Plot metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_f1_scores, label="Validation F1 Score")
    plt.plot(val_precisions, label="Validation Precision")
    plt.plot(val_recalls, label="Validation Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()

    plt.show()