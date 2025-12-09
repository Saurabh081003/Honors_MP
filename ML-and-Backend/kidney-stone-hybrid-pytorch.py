"""
Kidney Stone CT Classification - Hybrid Model (PyTorch + CUDA)
==============================================================
One-to-one conversion from TensorFlow/Keras version for local RTX 3060 training.

Architecture: ResNet50 + DenseNet121 + VGG16 ensemble with custom classification head
Task: 4-class classification (Normal, Cyst, Tumor, Stone)

Usage:
    conda create -n kidney_model python=3.10 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
    conda activate kidney_model
    pip install pandas scikit-learn matplotlib tqdm pillow
    python kidney_stone_hybrid_pytorch.py
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# === CONFIGURATION ===
class Config:
    # Paths - UPDATE THESE FOR YOUR SETUP
    # Point directly to folder containing Cyst/Normal/Stone/Tumor subfolders
    DATASET_PATH = "./datasets/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    OUTPUT_DIR = "./datasets/split_dataset"
    CHECKPOINT_DIR = "./datasets/checkpoints"
    
    # Set to True to delete and recreate split_dataset
    FORCE_RESPLIT = False
    
    # Training parameters
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 4
    NUM_WORKERS = 4  # Adjust based on CPU cores
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Early stopping
    PATIENCE = 10
    
    # Random seed for reproducibility
    SEED = 42

def set_seed(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === DATASET PREPARATION ===
def prepare_dataset(dataset_path, output_dir):
    """Split dataset into train/val/test (70/15/15) with stratification"""
    print("Preparing dataset...")
    
    # Valid image extensions
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # Get class folders
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Found class folders: {subdirs}")
    
    # Validate we have the expected classes
    expected = {'Cyst', 'Normal', 'Stone', 'Tumor'}
    found = {d.lower() for d in subdirs}
    if not expected.issubset(found):
        print(f"WARNING: Expected classes {expected}, found {found}")
    
    all_images = []
    all_labels = []
    
    for class_name in subdirs:
        class_path = os.path.join(dataset_path, class_name)
        img_count = 0
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            # Only process actual image files
            if os.path.isfile(img_path):
                ext = os.path.splitext(img_name)[1].lower()
                if ext in VALID_EXTENSIONS:
                    all_images.append(img_path)
                    all_labels.append(class_name)
                    img_count += 1
        print(f"  {class_name}: {img_count} images")
    
    if len(all_images) == 0:
        raise ValueError(
            f"No images found in {dataset_path}!\n"
            f"Expected structure: DATASET_PATH/ClassName/image.jpg\n"
            f"Found subdirs: {subdirs}"
        )
    
    print(f"\nTotal images found: {len(all_images)}")
    
    df = pd.DataFrame({"image": all_images, "label": all_labels})
    
    # Split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=Config.SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=Config.SEED
    )
    
    # Create directories
    for split in ['train', 'val', 'test']:
        for class_name in df['label'].unique():
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    # Copy images
    def copy_images(df_split, split_name):
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Copying {split_name}"):
            src_path = row['image']
            dest_dir = os.path.join(output_dir, split_name, row['label'])
            shutil.copy2(src_path, dest_dir)
    
    copy_images(train_df, 'train')
    copy_images(val_df, 'val')
    copy_images(test_df, 'test')
    
    print(f"\nDataset splitting completed!")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    return sorted(df['label'].unique())

# === DATA TRANSFORMS ===
def get_transforms():
    """Get train and validation/test transforms"""
    
    # Training augmentation (matches Keras version)
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),  # width_shift, height_shift
            shear=0.2,
            scale=(0.8, 1.2)  # zoom_range
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Validation/Test (only resize and normalize)
    val_test_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_test_transform

# === HYBRID MODEL ARCHITECTURE ===
class HybridKidneyModel(nn.Module):
    """
    Hybrid ensemble model combining ResNet50 + DenseNet121 + VGG16
    
    Feature dimensions:
        - ResNet50: 2048
        - DenseNet121: 1024  
        - VGG16: 512
        - Combined: 3584
    """
    
    def __init__(self, num_classes=4, pretrained=True, freeze_backbones=True):
        super(HybridKidneyModel, self).__init__()
        
        # === BACKBONE NETWORKS ===
        # ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove FC
        
        # DenseNet121
        self.densenet = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        self.densenet_features = self.densenet.features
        
        # VGG16
        self.vgg = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
        self.vgg_features = self.vgg.features
        
        # Freeze backbones if specified
        if freeze_backbones:
            for model in [self.resnet_features, self.densenet_features, self.vgg_features]:
                for param in model.parameters():
                    param.requires_grad = False
        
        # === POOLING LAYERS ===
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # === CLASSIFICATION HEAD ===
        # Combined features: 2048 (ResNet) + 1024 (DenseNet) + 512 (VGG) = 3584
        self.classifier = nn.Sequential(
            # FC1
            nn.Linear(3584, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            
            # FC2
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            
            # FC3
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            # FC4
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # Output
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # ResNet50 features
        resnet_out = self.resnet_features(x)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)  # Flatten: [B, 2048]
        
        # DenseNet121 features
        densenet_out = self.densenet_features(x)
        densenet_out = self.global_pool(densenet_out)
        densenet_out = densenet_out.view(densenet_out.size(0), -1)  # [B, 1024]
        
        # VGG16 features
        vgg_out = self.vgg_features(x)
        vgg_out = self.global_pool(vgg_out)
        vgg_out = vgg_out.view(vgg_out.size(0), -1)  # [B, 512]
        
        # Concatenate features
        combined = torch.cat([resnet_out, densenet_out, vgg_out], dim=1)  # [B, 3584]
        
        # Classification
        output = self.classifier(combined)
        
        return output

# === TRAINING UTILITIES ===
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            self.counter = 0
    
    def restore(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class MetricTracker:
    """Track and compute metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss = 0
        self.correct = 0
        self.total = 0
        self.all_preds = []
        self.all_labels = []
    
    def update(self, loss, preds, labels):
        self.loss += loss
        self.correct += (preds == labels).sum().item()
        self.total += labels.size(0)
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
    
    def get_metrics(self, num_batches):
        accuracy = self.correct / self.total
        avg_loss = self.loss / num_batches
        return avg_loss, accuracy

# === TRAINING LOOP ===
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    tracker = MetricTracker()
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        tracker.update(loss.item(), preds, labels)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return tracker.get_metrics(len(dataloader))


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    tracker = MetricTracker()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            tracker.update(loss.item(), preds, labels)
    
    return tracker.get_metrics(len(dataloader))


def calculate_precision_recall_auc(model, dataloader, device, num_classes):
    """Calculate precision, recall, and AUC for test evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Calculate AUC (one-vs-rest)
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        print(f"AUC (OvR): {auc:.4f}")
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
        auc = None
    
    return all_preds, all_labels, auc


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    axes[0].plot(history['train_acc'], label='Training Accuracy')
    axes[0].plot(history['val_acc'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history['train_loss'], label='Training Loss')
    axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Training history saved to {save_path}")


# === MAIN TRAINING FUNCTION ===
def main():
    # Set seed
    set_seed(Config.SEED)
    
    # Print device info
    print(f"\n{'='*60}")
    print(f"Kidney Stone Classification - Hybrid Model (PyTorch)")
    print(f"{'='*60}")
    print(f"Device: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    
    # Check if dataset needs to be prepared
    train_dir = os.path.join(Config.OUTPUT_DIR, 'train')
    
    # Force resplit if configured or if split looks corrupted
    if os.path.exists(Config.OUTPUT_DIR):
        existing_classes = os.listdir(train_dir) if os.path.exists(train_dir) else []
        expected_classes = {'Cyst', 'Normal', 'Stone', 'Tumor'}
        is_valid_split = expected_classes.issubset(set(existing_classes)) or \
                         set(c.lower() for c in existing_classes) >= {c.lower() for c in expected_classes}
        
        if Config.FORCE_RESPLIT or not is_valid_split:
            print(f"Removing invalid/old split_dataset (found classes: {existing_classes})...")
            shutil.rmtree(Config.OUTPUT_DIR)
    
    if not os.path.exists(train_dir):
        if not os.path.exists(Config.DATASET_PATH):
            print(f"ERROR: Dataset not found at {Config.DATASET_PATH}")
            print("Please update Config.DATASET_PATH to point to your dataset.")
            return
        class_names = prepare_dataset(Config.DATASET_PATH, Config.OUTPUT_DIR)
    else:
        print("Using existing split dataset...")
        class_names = sorted([d for d in os.listdir(train_dir) 
                              if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"Classes: {class_names}")
    
    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms()
    
    # Create datasets
    train_dataset = ImageFolder(
        os.path.join(Config.OUTPUT_DIR, 'train'),
        transform=train_transform
    )
    val_dataset = ImageFolder(
        os.path.join(Config.OUTPUT_DIR, 'val'),
        transform=val_test_transform
    )
    test_dataset = ImageFolder(
        os.path.join(Config.OUTPUT_DIR, 'test'),
        transform=val_test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_dataset.targets),
        y=train_dataset.targets
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(Config.DEVICE)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    
    # Create model
    print("\nBuilding Hybrid Model...")
    model = HybridKidneyModel(
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        freeze_backbones=True
    ).to(Config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE
    )
    
    # Learning rate scheduler (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=5,
        min_lr=1e-7
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    
    # === TRAINING LOOP ===
    print(f"\n{'='*60}")
    print("Starting Training...")
    print(f"{'='*60}\n")
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, Config.DEVICE
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(Config.CHECKPOINT_DIR, 'best_hybrid_model.pth'))
            print(f"✓ New best model saved! (Val Acc: {val_acc:.4f})")
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best weights
    early_stopping.restore(model)
    
    # === EVALUATION ===
    print(f"\n{'='*60}")
    print("Evaluating on Test Set...")
    print(f"{'='*60}\n")
    
    # Load best model
    checkpoint = torch.load(os.path.join(Config.CHECKPOINT_DIR, 'best_hybrid_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_acc = validate(model, test_loader, criterion, Config.DEVICE)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Detailed metrics
    calculate_precision_recall_auc(model, test_loader, Config.DEVICE, Config.NUM_CLASSES)
    
    # Plot training history
    plot_training_history(history)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {Config.CHECKPOINT_DIR}/best_hybrid_model.pth")
    print(f"{'='*60}")


# === PREDICTION FUNCTION ===
def predict_image(model, image_path, class_names, device=Config.DEVICE):
    """Predict class for a single image"""
    from PIL import Image
    
    _, transform = get_transforms()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score


def load_model_for_inference(checkpoint_path, device=Config.DEVICE):
    """Load trained model for inference"""
    model = HybridKidneyModel(num_classes=Config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    main()