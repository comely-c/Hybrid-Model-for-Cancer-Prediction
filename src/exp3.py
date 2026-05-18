# ============================================================
# FYP — SIPaKMeD Image-only Cervical Cell Classification Pipeline
#
# Thesis core:
#   1) Single deep learning models
#      - Custom CNN
#      - MobileNetV2
#      - ResNet18
#   2) Hybrid models
#      - Custom CNN features + XGBoost
#      - MobileNetV2 features + XGBoost
#      - ResNet18 features + XGBoost
#   3) Explainable AI
#      - Grad-CAM
#      - t-SNE feature visualization
#   4) Lightweight/performance comparison
#      - Accuracy, Precision, Recall, F1, AUC, PR-AUC
#      - Training time, parameter count, model size
#      - Inference time/image, FPS, accuracy per million parameters
#   5) Extra thesis analysis
#      - Ablation study: augmentation, sampler, CNN vs CNN+XGBoost
#      - Best single / best hybrid / best lightweight model summary
#      - Extended Grad-CAM: correct examples by class + misclassified examples
#
# Recommended for local Windows/VSCode running.
# ============================================================

import os
import time
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from xgboost import XGBClassifier


# ============================================================
# 1. BASIC SETTINGS
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # deterministic=True is safer for reproducibility but can be slower.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Change these paths if needed.
DATA_ROOT = Path(r"E:\sipakmed")
OUTPUT_DIR = Path(r"D:\vscode\Code\FYP\outputs\exp3_image_only")
CHECKPOINT_DIR = Path(r"E:\FYP\checkpoints_exp3_image_only")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate",
]

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 20
PATIENCE = 6
WEIGHT_DECAY = 1e-4

LR_CUSTOM_CNN = 1e-3
LR_MOBILENET = 1e-4
LR_RESNET = 1e-4
LR_EFFICIENTNET = 1e-4

# EfficientNet-B0 is useful if you want one more lightweight pretrained comparison.
# Keep False if you want a faster final run.
INCLUDE_EFFICIENTNET_B0 = False

# Ablation runs make the thesis stronger, but they add extra training time.
# These are CustomCNN-only ablations, so they are much lighter than retraining all models.
RUN_ABLATION_STUDY = True
ABLATION_EPOCHS = 8
ABLATION_PATIENCE = 3

# Lightweight hyperparameter tuning for Phase 3.
# Default is True, but the search is intentionally small to avoid excessive runtime.
RUN_HYPERPARAMETER_TUNING = True
TUNING_EPOCHS = 6
TUNING_PATIENCE = 2

BINARY_MAP = {
    "im_Dyskeratotic": 1,
    "im_Koilocytotic": 1,
    "im_Metaplastic": 0,
    "im_Parabasal": 0,
    "im_Superficial-Intermediate": 0,
}


def out(fname):
    return OUTPUT_DIR / fname


def safe_save_fig(fname, **kwargs):
    plt.savefig(out(fname), bbox_inches="tight", **kwargs)


def safe_save_csv(df, fname):
    path = out(fname)
    df.to_csv(path, index=True)
    return path


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def get_file_size_mb(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 ** 2)


# ============================================================
# 2. DATASET AND TRANSFORMS
# ============================================================

def build_transforms(phase="train", use_augmentation=True):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if phase == "train" and use_augmentation:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])


class SIPaKMeDDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []

        for label, class_name in enumerate(CLASS_NAMES):
            possible_dirs = [
                self.data_root / class_name / class_name,
                self.data_root / class_name,
            ]

            class_dir = None
            for d in possible_dirs:
                if d.exists():
                    class_dir = d
                    break

            if class_dir is None:
                print(f"[WARN] Missing folder for class: {class_name}")
                continue

            image_files = sorted(list(class_dir.glob("*.bmp")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")))
            for img_path in image_files:
                self.samples.append((img_path, label))

        print(f"Total images found: {len(self.samples)}")
        counts = Counter([label for _, label in self.samples])
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {i}: {name} = {counts.get(i, 0)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not load image {img_path}: {e}")
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)
        return image, label


class TransformSubset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, label = self.base_dataset.samples[real_idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)
        return image, label


def load_datasets(data_root, use_augmentation=True, use_sampler=True, save_split=True):
    print_section("DATA LOADING AND PREPROCESSING")

    full_dataset = SIPaKMeDDataset(data_root, transform=None)
    if len(full_dataset) == 0:
        raise RuntimeError(f"No images found under {data_root}. Please check DATA_ROOT.")

    labels = [label for _, label in full_dataset.samples]
    indices = list(range(len(full_dataset)))

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    print(f"Train size: {len(train_idx)}")
    print(f"Validation size: {len(val_idx)}")
    print(f"Test size: {len(test_idx)}")

    train_dataset = TransformSubset(full_dataset, train_idx, build_transforms("train", use_augmentation=use_augmentation))
    val_dataset = TransformSubset(full_dataset, val_idx, build_transforms("val"))
    test_dataset = TransformSubset(full_dataset, test_idx, build_transforms("test"))

    train_labels = [labels[i] for i in train_idx]
    train_counts = Counter(train_labels)
    sample_weights = [1.0 / train_counts[labels[i]] for i in train_idx]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    ) if use_sampler else None

    # num_workers=0 is safest on Windows/VSCode.
    # If your local machine is stable, you may try num_workers=2.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    if save_split:
        split_df = pd.DataFrame({
            "split": ["train"] * len(train_idx) + ["val"] * len(val_idx) + ["test"] * len(test_idx),
            "index": train_idx + val_idx + test_idx,
            "label": [labels[i] for i in train_idx + val_idx + test_idx],
            "class_name": [CLASS_NAMES[labels[i]] for i in train_idx + val_idx + test_idx],
        })
        safe_save_csv(split_df, "dataset_split_70_15_15.csv")

    return full_dataset, train_loader, val_loader, test_loader


# ============================================================
# 3. MODEL ARCHITECTURES
# ============================================================

class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_p=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features.unsqueeze(-1).unsqueeze(-1))

    def extract_features(self, x):
        x = self.features(x)
        x = self.gap(x)
        return x.squeeze(-1).squeeze(-1)


class MobileNetV2WithFeatures(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout_p=0.3):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)

    def extract_features(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1)


class ResNet18WithFeatures(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout_p=0.3):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        in_features = base.fc.in_features
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)

    def extract_features(self, x):
        x = self.backbone(x)
        return x.squeeze(-1).squeeze(-1)


class EfficientNetB0WithFeatures(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout_p=0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)

    def extract_features(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1)


# ============================================================
# 4. TRAINING AND EVALUATION
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_true, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_true), np.array(all_preds), np.vstack(all_probs)


def measure_inference_speed(model, loader, device, warmup_batches=2, max_batches=10):
    model.eval()
    total_images = 0
    start = None

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if batch_idx == warmup_batches:
                start = time.time()
                total_images = 0
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if batch_idx >= warmup_batches:
                total_images += images.size(0)
            if batch_idx + 1 >= warmup_batches + max_batches:
                break

    elapsed = time.time() - start if start is not None else 0.0
    if elapsed <= 0 or total_images == 0:
        return 0.0, 0.0
    inference_ms_per_image = 1000 * elapsed / total_images
    fps = total_images / elapsed
    return round(inference_ms_per_image, 4), round(fps, 2)


def compute_metrics(y_true, y_pred, y_prob):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    try:
        auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan
    try:
        pr_auc = average_precision_score(y_true_bin, y_prob, average="macro")
    except Exception:
        pr_auc = np.nan

    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "AUC": round(auc, 4) if not np.isnan(auc) else 0.0,
        "PR-AUC": round(pr_auc, 4) if not np.isnan(pr_auc) else 0.0,
    }


def train_model(model, train_loader, val_loader, model_name, lr, device, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    start_train = time.time()
    best_ckpt_path = CHECKPOINT_DIR / f"{model_name}_best.pth"

    print(f"\nTraining {model_name}...")
    for epoch in range(1, num_epochs + 1):
        start_epoch = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} | "
            f"Time={time.time() - start_epoch:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), best_ckpt_path)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    training_time = time.time() - start_train
    model_size_mb = get_file_size_mb(best_ckpt_path)
    return model, history, training_time, model_size_mb


def extract_embeddings(model, loader, device):
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            features = model.extract_features(images)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.vstack(all_features), np.array(all_labels)


# ============================================================
# 5. HYBRID MODEL: DEEP FEATURES + XGBOOST
# ============================================================

def train_xgboost_hybrid(model_name, train_features, train_labels, test_features, test_labels):
    print(f"\nHybrid: {model_name} features + XGBoost")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)

    train_counts = Counter(train_labels)
    n_train = len(train_labels)
    sample_weights = np.array([n_train / (NUM_CLASSES * train_counts[label]) for label in train_labels])

    param_grid = {
        "n_estimators": [100],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    start = time.time()
    grid = GridSearchCV(
        estimator=XGBClassifier(
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=1,
            tree_method="hist",
        ),
        param_grid=param_grid,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        n_jobs=1,
        verbose=0,
    )
    grid.fit(X_train, train_labels, sample_weight=sample_weights)
    training_time = time.time() - start

    best_xgb = grid.best_estimator_
    probs = best_xgb.predict_proba(X_test)
    preds = probs.argmax(axis=1)
    metrics = compute_metrics(test_labels, preds, probs)
    metrics["Training_Time_sec"] = round(training_time, 2)
    metrics["Best_Params"] = str(grid.best_params_)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV macro F1: {grid.best_score_:.4f}")
    print(metrics)

    return best_xgb, scaler, preds, probs, metrics


# ============================================================
# 6. GRAD-CAM
# ============================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, device, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# 7. VISUALISATION HELPERS
# ============================================================

def plot_training_curves(histories):
    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        plt.plot(hist["val_acc"], marker="o", label=f"{name} Val Acc")
    plt.title("Validation Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    safe_save_fig("viz_validation_accuracy_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.replace("im_", "") for c in CLASS_NAMES],
        yticklabels=[c.replace("im_", "") for c in CLASS_NAMES],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    safe_save_fig(filename, dpi=150)
    plt.close()


def plot_metric_bar(results_df, filename):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC", "PR-AUC"]
    results_df[metrics].plot(kind="bar", figsize=(13, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    safe_save_fig(filename, dpi=150)
    plt.close()


def plot_efficiency(results_df):
    efficiency_cols = ["Training_Time_sec", "Trainable_Params", "Model_Size_MB", "Inference_ms_per_image", "FPS", "Accuracy_per_Million_Params"]
    existing_cols = [c for c in efficiency_cols if c in results_df.columns]
    if not existing_cols:
        return
    efficiency_df = results_df[existing_cols].copy()
    for col in existing_cols:
        efficiency_df[col] = pd.to_numeric(efficiency_df[col], errors="coerce")
    efficiency_df.plot(kind="bar", figsize=(13, 6))
    plt.title("Computational Efficiency Comparison")
    plt.ylabel("Value")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    safe_save_fig("viz_efficiency_comparison.png", dpi=150)
    plt.close()


def plot_tsne(features, labels, model_name):
    n_samples = min(len(features), 1000)
    sample_idx = np.random.choice(len(features), n_samples, replace=False)
    features_sample = features[sample_idx]
    labels_sample = labels[sample_idx]

    features_scaled = StandardScaler().fit_transform(features_sample)
    perplexity = min(30, max(5, n_samples // 10))
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity, learning_rate="auto", init="pca")
    features_2d = tsne.fit_transform(features_scaled)

    plt.figure(figsize=(9, 7))
    for class_idx in range(NUM_CLASSES):
        mask = labels_sample == class_idx
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], s=25, alpha=0.7, label=CLASS_NAMES[class_idx].replace("im_", ""))
    plt.title(f"t-SNE of {model_name} Deep Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    safe_save_fig(f"viz_tsne_{model_name}.png", dpi=150)
    plt.close()


def collect_gradcam_examples(model, loader, device, max_wrong=2):
    model.eval()
    correct_by_class = {}
    wrong_examples = []

    with torch.no_grad():
        for images, labels in loader:
            images_device = images.to(device, non_blocking=True)
            outputs = model(images_device)
            preds = outputs.argmax(dim=1).cpu()

            for i in range(images.size(0)):
                true_label = int(labels[i].item())
                pred_label = int(preds[i].item())
                example = (images[i].clone(), true_label, pred_label)

                if true_label == pred_label and true_label not in correct_by_class:
                    correct_by_class[true_label] = example
                elif true_label != pred_label and len(wrong_examples) < max_wrong:
                    wrong_examples.append(example)

            if len(correct_by_class) == NUM_CLASSES and len(wrong_examples) >= max_wrong:
                break

    examples = [correct_by_class[i] for i in range(NUM_CLASSES) if i in correct_by_class]
    examples.extend(wrong_examples)
    return examples


def plot_gradcam(model, model_name, target_layer, test_loader, device, max_wrong=2):
    gradcam = GradCAM(model, target_layer)
    examples = collect_gradcam_examples(model, test_loader, device, max_wrong=max_wrong)

    if len(examples) == 0:
        print(f"[WARN] No Grad-CAM examples found for {model_name}")
        gradcam.close()
        return

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    n = len(examples)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for i, (img_tensor, true_label, pred_label_before) in enumerate(examples):
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np * std + mean, 0, 1)

        cam, pred_class = gradcam.generate(img_tensor, device)
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
        cam_resized = np.array(cam_img) / 255.0
        status = "Correct" if true_label == pred_class else "Wrong"

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(
            f"{status}\nTrue: {CLASS_NAMES[true_label].replace('im_', '')}\n"
            f"Pred: {CLASS_NAMES[pred_class].replace('im_', '')}",
            fontsize=9,
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(img_np)
        axes[1, i].imshow(cam_resized, cmap="jet", alpha=0.45)
        axes[1, i].set_title("Grad-CAM", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle(f"Extended Grad-CAM — {model_name}: one correct sample per class + misclassified samples")
    plt.tight_layout()
    safe_save_fig(f"viz_gradcam_extended_{model_name}.png", dpi=150)
    plt.close()
    gradcam.close()


def plot_macro_roc(roc_models):
    plt.figure(figsize=(9, 7))
    for model_name, (y_true, y_prob) in roc_models.items():
        y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        try:
            fpr, tpr = {}, {}
            for i in range(NUM_CLASSES):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(NUM_CLASSES):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= NUM_CLASSES
            auc_score = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
            plt.plot(all_fpr, mean_tpr, linewidth=2, label=f"{model_name} AUC={auc_score:.3f}")
        except Exception:
            pass

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title("Macro-average ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    safe_save_fig("viz_macro_roc_curves.png", dpi=150)
    plt.close()


def save_classification_reports(model_outputs):
    report_rows = []
    f1_report = {}
    for model_name, (y_true, y_pred) in model_outputs.items():
        report = classification_report(
            y_true,
            y_pred,
            target_names=[c.replace("im_", "") for c in CLASS_NAMES],
            output_dict=True,
            zero_division=0,
        )
        for class_name in [c.replace("im_", "") for c in CLASS_NAMES]:
            report_rows.append({
                "Model": model_name,
                "Class": class_name,
                "Precision": report[class_name]["precision"],
                "Recall": report[class_name]["recall"],
                "F1": report[class_name]["f1-score"],
                "Support": report[class_name]["support"],
            })
        f1_report[model_name] = {class_name: report[class_name]["f1-score"] for class_name in [c.replace("im_", "") for c in CLASS_NAMES]}

    report_df = pd.DataFrame(report_rows)
    safe_save_csv(report_df, "classification_reports_by_class.csv")

    f1_df = pd.DataFrame(f1_report).T
    plt.figure(figsize=(11, 5))
    sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Per-class F1-score Heatmap")
    plt.xlabel("Class")
    plt.ylabel("Model")
    plt.xticks(rotation=45, ha="right")
    safe_save_fig("viz_per_class_f1_heatmap.png", dpi=150)
    plt.close()
    safe_save_csv(f1_df, "per_class_f1_scores.csv")


def compute_binary_metrics(y_true_5, y_pred_5, y_prob_5):
    binary_label_map = np.array([BINARY_MAP[class_name] for class_name in CLASS_NAMES])
    y_true_bin = binary_label_map[y_true_5]
    y_pred_bin = binary_label_map[y_pred_5]
    abnormal_prob = y_prob_5[:, 0] + y_prob_5[:, 1]
    try:
        binary_auc = roc_auc_score(y_true_bin, abnormal_prob)
    except Exception:
        binary_auc = np.nan
    return {
        "Binary_Accuracy": round(accuracy_score(y_true_bin, y_pred_bin), 4),
        "Binary_Precision": round(precision_score(y_true_bin, y_pred_bin, zero_division=0), 4),
        "Binary_Recall": round(recall_score(y_true_bin, y_pred_bin, zero_division=0), 4),
        "Binary_F1": round(f1_score(y_true_bin, y_pred_bin, zero_division=0), 4),
        "Binary_AUC": round(binary_auc, 4) if not np.isnan(binary_auc) else 0.0,
    }



def save_best_model_summary(final_results):
    df = final_results.copy()
    df["Accuracy_num"] = pd.to_numeric(df["Accuracy"], errors="coerce")
    df["F1_num"] = pd.to_numeric(df["F1"], errors="coerce")
    df["Params_num"] = pd.to_numeric(df.get("Trainable_Params"), errors="coerce")

    summary_rows = []

    single_df = df[df["Model_Type"] == "Single"]
    hybrid_df = df[df["Model_Type"] == "Hybrid"]

    if len(single_df) > 0:
        best_single = single_df.sort_values(["F1_num", "Accuracy_num"], ascending=False).iloc[0]
        summary_rows.append({"Category": "Best single model", "Model": best_single.name, "Selection_Rule": "Highest macro F1, then accuracy"})

    if len(hybrid_df) > 0:
        best_hybrid = hybrid_df.sort_values(["F1_num", "Accuracy_num"], ascending=False).iloc[0]
        summary_rows.append({"Category": "Best hybrid model", "Model": best_hybrid.name, "Selection_Rule": "Highest macro F1, then accuracy"})

    lightweight_df = single_df.dropna(subset=["Params_num"]).sort_values(["F1_num", "Params_num"], ascending=[False, True])
    if len(lightweight_df) > 0:
        best_light = lightweight_df.iloc[0]
        summary_rows.append({"Category": "Best lightweight model", "Model": best_light.name, "Selection_Rule": "High macro F1 with lower trainable parameters"})

    summary_df = pd.DataFrame(summary_rows)
    safe_save_csv(summary_df, "best_model_summary.csv")
    print("\nBest Model Summary:")
    print(summary_df)
    return summary_df


def run_ablation_study(test_loader, device):
    if not RUN_ABLATION_STUDY:
        return pd.DataFrame()

    print_section("PHASE 4 — ABLATION STUDY")

    ablation_settings = {
        "CNN_no_augmentation_with_sampler": {"use_augmentation": False, "use_sampler": True},
        "CNN_with_augmentation_no_sampler": {"use_augmentation": True, "use_sampler": False},
        "CNN_with_augmentation_with_sampler": {"use_augmentation": True, "use_sampler": True},
    }

    rows = {}
    criterion_eval = nn.CrossEntropyLoss()

    for ablation_name, cfg in ablation_settings.items():
        print(f"\nRunning ablation: {ablation_name}")
        _, train_loader_ab, val_loader_ab, test_loader_ab = load_datasets(
            DATA_ROOT,
            use_augmentation=cfg["use_augmentation"],
            use_sampler=cfg["use_sampler"],
            save_split=False,
        )

        model = CustomCNN(num_classes=NUM_CLASSES, dropout_p=0.5)
        model, _, train_time, model_size_mb = train_model(
            model=model,
            train_loader=train_loader_ab,
            val_loader=val_loader_ab,
            model_name=f"ablation_{ablation_name}",
            lr=LR_CUSTOM_CNN,
            device=device,
            num_epochs=ABLATION_EPOCHS,
            patience=ABLATION_PATIENCE,
        )
        _, _, y_true, y_pred, y_prob = evaluate_model(model, test_loader_ab, criterion_eval, device)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["Training_Time_sec"] = round(train_time, 2)
        metrics["Model_Size_MB"] = round(model_size_mb, 2)
        metrics["Use_Augmentation"] = cfg["use_augmentation"]
        metrics["Use_WeightedRandomSampler"] = cfg["use_sampler"]
        rows[ablation_name] = metrics

    ablation_df = pd.DataFrame(rows).T
    safe_save_csv(ablation_df, "ablation_study_results.csv")

    plot_cols = ["Accuracy", "Recall", "F1", "AUC"]
    ablation_df[plot_cols].plot(kind="bar", figsize=(11, 6))
    plt.title("Ablation Study: Augmentation and WeightedRandomSampler")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    safe_save_fig("viz_ablation_study.png", dpi=150)
    plt.close()

    print("\nAblation Study Results:")
    print(ablation_df)
    return ablation_df


# ============================================================
# 8. ENSEMBLE, OPTIMIZATION, AND REPORT HELPERS
# ============================================================

def average_probs(model_names, prob_store, weights=None):
    probs = [prob_store[name] for name in model_names]
    if weights is None:
        return np.mean(probs, axis=0)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    result = np.zeros_like(probs[0])
    for w, p in zip(weights, probs):
        result += w * p
    return result


def compute_validation_weights(model_names, val_true_store, val_prob_store):
    scores = []
    y_true = val_true_store[model_names[0]]
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    for name in model_names:
        probs = val_prob_store[name]
        try:
            score = roc_auc_score(y_true_bin, probs, multi_class="ovr", average="macro")
        except Exception:
            preds = probs.argmax(axis=1)
            score = f1_score(y_true, preds, average="macro", zero_division=0)
        scores.append(max(score, 1e-6))

    weights = np.array(scores) / np.sum(scores)
    return weights


def build_ensemble_results(model_names, test_true, test_prob_store, val_true_store, val_prob_store):
    print_section("PHASE 2 — ENSEMBLE LEARNING")

    ensemble_results = {}
    ensemble_outputs = {}
    ensemble_probs_for_curves = {}

    # Ensemble 1: MobileNetV2 + ResNet18 soft voting
    ensemble1_names = ["MobileNetV2", "ResNet18"]
    e1_probs = average_probs(ensemble1_names, test_prob_store)
    e1_preds = e1_probs.argmax(axis=1)
    e1_name = "Ensemble1_SoftVoting_MobileNetV2_ResNet18"
    e1_metrics = compute_metrics(test_true, e1_preds, e1_probs)
    e1_metrics["Model_Type"] = "Ensemble"
    e1_metrics["Ensemble_Method"] = "Soft voting"
    e1_metrics["Base_Models"] = "+".join(ensemble1_names)
    ensemble_results[e1_name] = e1_metrics
    ensemble_outputs[e1_name] = (test_true, e1_preds)
    ensemble_probs_for_curves[e1_name] = (test_true, e1_probs)
    print(f"\n{e1_name}:")
    print(e1_metrics)

    # Ensemble 2: CustomCNN + MobileNetV2 + ResNet18 weighted voting
    ensemble2_names = ["CustomCNN", "MobileNetV2", "ResNet18"]
    e2_weights = compute_validation_weights(ensemble2_names, val_true_store, val_prob_store)
    e2_probs = average_probs(ensemble2_names, test_prob_store, weights=e2_weights)
    e2_preds = e2_probs.argmax(axis=1)
    e2_name = "Ensemble2_WeightedVoting_CustomCNN_MobileNetV2_ResNet18"
    e2_metrics = compute_metrics(test_true, e2_preds, e2_probs)
    e2_metrics["Model_Type"] = "Ensemble"
    e2_metrics["Ensemble_Method"] = "Validation-AUC weighted voting"
    e2_metrics["Base_Models"] = "+".join(ensemble2_names)
    e2_metrics["Weights"] = str({name: round(float(w), 4) for name, w in zip(ensemble2_names, e2_weights)})
    ensemble_results[e2_name] = e2_metrics
    ensemble_outputs[e2_name] = (test_true, e2_preds)
    ensemble_probs_for_curves[e2_name] = (test_true, e2_probs)
    print(f"\n{e2_name}:")
    print(e2_metrics)

    # Ensemble 3: stacking ensemble using validation predictions as meta-training data
    stack_names = ["CustomCNN", "MobileNetV2", "ResNet18"]
    X_meta_val = np.hstack([val_prob_store[name] for name in stack_names])
    y_meta_val = val_true_store[stack_names[0]]
    X_meta_test = np.hstack([test_prob_store[name] for name in stack_names])

    stacker = LogisticRegression(
        max_iter=2000,
        multi_class="auto",
        class_weight="balanced",
        random_state=SEED,
    )
    stack_start = time.time()
    stacker.fit(X_meta_val, y_meta_val)
    stacking_time = time.time() - stack_start
    stack_probs = stacker.predict_proba(X_meta_test)
    stack_preds = stack_probs.argmax(axis=1)
    stack_name = "Ensemble3_Stacking_CustomCNN_MobileNetV2_ResNet18"
    stack_metrics = compute_metrics(test_true, stack_preds, stack_probs)
    stack_metrics["Model_Type"] = "Ensemble"
    stack_metrics["Ensemble_Method"] = "Stacking with LogisticRegression meta-classifier"
    stack_metrics["Base_Models"] = "+".join(stack_names)
    stack_metrics["Training_Time_sec"] = round(stacking_time, 4)
    ensemble_results[stack_name] = stack_metrics
    ensemble_outputs[stack_name] = (test_true, stack_preds)
    ensemble_probs_for_curves[stack_name] = (test_true, stack_probs)
    print(f"\n{stack_name}:")
    print(stack_metrics)

    ensemble_df = pd.DataFrame(ensemble_results).T
    safe_save_csv(ensemble_df, "phase2_ensemble_results.csv")
    return ensemble_df, ensemble_outputs, ensemble_probs_for_curves


def plot_macro_pr(pr_models):
    plt.figure(figsize=(9, 7))
    for model_name, (y_true, y_prob) in pr_models.items():
        y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        try:
            precision, recall = {}, {}
            for i in range(NUM_CLASSES):
                precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            all_recall = np.unique(np.concatenate([recall[i] for i in range(NUM_CLASSES)]))
            mean_precision = np.zeros_like(all_recall)
            for i in range(NUM_CLASSES):
                # precision_recall_curve returns recall in descending order, reverse for interpolation
                mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
            mean_precision /= NUM_CLASSES
            pr_auc = average_precision_score(y_true_bin, y_prob, average="macro")
            plt.plot(all_recall, mean_precision, linewidth=2, label=f"{model_name} AP={pr_auc:.3f}")
        except Exception:
            pass
    plt.title("Macro-average Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    safe_save_fig("viz_macro_pr_curves.png", dpi=150)
    plt.close()


def run_hyperparameter_tuning(device):
    if not RUN_HYPERPARAMETER_TUNING:
        return pd.DataFrame()

    print_section("PHASE 3 — OPTIMIZATION: HYPERPARAMETER TUNING")
    print("This is a lightweight CustomCNN tuning run for thesis optimization analysis.")

    tuning_grid = [
        {"lr": 1e-3, "dropout": 0.3},
        {"lr": 1e-3, "dropout": 0.5},
        {"lr": 5e-4, "dropout": 0.5},
    ]

    rows = {}
    criterion_eval = nn.CrossEntropyLoss()

    for cfg in tuning_grid:
        run_name = f"CustomCNN_lr{cfg['lr']}_dropout{cfg['dropout']}".replace(".", "p")
        print(f"\nTuning run: {run_name}")
        _, train_loader_tune, val_loader_tune, test_loader_tune = load_datasets(
            DATA_ROOT,
            use_augmentation=True,
            use_sampler=True,
            save_split=False,
        )
        model = CustomCNN(num_classes=NUM_CLASSES, dropout_p=cfg["dropout"])
        model, _, train_time, model_size_mb = train_model(
            model=model,
            train_loader=train_loader_tune,
            val_loader=val_loader_tune,
            model_name=f"tuning_{run_name}",
            lr=cfg["lr"],
            device=device,
            num_epochs=TUNING_EPOCHS,
            patience=TUNING_PATIENCE,
        )
        _, _, y_true, y_pred, y_prob = evaluate_model(model, test_loader_tune, criterion_eval, device)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["Learning_Rate"] = cfg["lr"]
        metrics["Dropout"] = cfg["dropout"]
        metrics["Training_Time_sec"] = round(train_time, 2)
        metrics["Model_Size_MB"] = round(model_size_mb, 2)
        rows[run_name] = metrics

    tuning_df = pd.DataFrame(rows).T
    safe_save_csv(tuning_df, "phase3_hyperparameter_tuning_results.csv")
    print("\nHyperparameter Tuning Results:")
    print(tuning_df)
    return tuning_df


def save_ensemble_vs_single_summary(single_df, ensemble_df):
    combined = pd.concat([single_df, ensemble_df], axis=0, sort=False)
    combined["F1_num"] = pd.to_numeric(combined["F1"], errors="coerce")
    combined["Accuracy_num"] = pd.to_numeric(combined["Accuracy"], errors="coerce")

    best_single = combined[combined["Model_Type"] == "Single"].sort_values(["F1_num", "Accuracy_num"], ascending=False).iloc[0]
    best_ensemble = combined[combined["Model_Type"] == "Ensemble"].sort_values(["F1_num", "Accuracy_num"], ascending=False).iloc[0]

    summary = pd.DataFrame([
        {
            "Category": "Best single model",
            "Model": best_single.name,
            "Accuracy": best_single["Accuracy"],
            "F1": best_single["F1"],
            "AUC": best_single["AUC"],
        },
        {
            "Category": "Best ensemble model",
            "Model": best_ensemble.name,
            "Accuracy": best_ensemble["Accuracy"],
            "F1": best_ensemble["F1"],
            "AUC": best_ensemble["AUC"],
        },
        {
            "Category": "Ensemble improvement over best single",
            "Model": f"{best_ensemble.name} vs {best_single.name}",
            "Accuracy": round(float(best_ensemble["Accuracy"]) - float(best_single["Accuracy"]), 4),
            "F1": round(float(best_ensemble["F1"]) - float(best_single["F1"]), 4),
            "AUC": round(float(best_ensemble["AUC"]) - float(best_single["AUC"]), 4),
        },
    ])
    safe_save_csv(summary, "ensemble_vs_single_summary.csv")
    print("\nEnsemble vs Single Summary:")
    print(summary)
    return summary


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset, train_loader, val_loader, test_loader = load_datasets(
        DATA_ROOT,
        use_augmentation=True,
        use_sampler=True,
        save_split=True,
    )
    criterion_eval = nn.CrossEntropyLoss()

    single_model_configs = {
        "CustomCNN": (CustomCNN(num_classes=NUM_CLASSES, dropout_p=0.5), LR_CUSTOM_CNN),
        "MobileNetV2": (MobileNetV2WithFeatures(num_classes=NUM_CLASSES, pretrained=True), LR_MOBILENET),
        "ResNet18": (ResNet18WithFeatures(num_classes=NUM_CLASSES, pretrained=True), LR_RESNET),
    }

    if INCLUDE_EFFICIENTNET_B0:
        single_model_configs["EfficientNetB0"] = (
            EfficientNetB0WithFeatures(num_classes=NUM_CLASSES, pretrained=True),
            LR_EFFICIENTNET,
        )

    trained_models = {}
    histories = {}
    single_results = {}
    single_outputs = {}
    val_true_store = {}
    val_prob_store = {}
    test_prob_store = {}
    curves_store = {}
    feature_store = {}

    # ------------------------------------------------------------
    # PHASE 1 — SINGLE DEEP LEARNING MODELS
    # ------------------------------------------------------------
    print_section("PHASE 1 — SINGLE DEEP LEARNING MODELS")

    for model_name, (model, lr) in single_model_configs.items():
        total_params = count_total_params(model)
        trainable_params = count_trainable_params(model)

        model, history, train_time, model_size_mb = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=f"phase1_single_{model_name}",
            lr=lr,
            device=device,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
        )

        _, _, val_true, val_pred, val_prob = evaluate_model(model, val_loader, criterion_eval, device)
        _, _, y_true, y_pred, y_prob = evaluate_model(model, test_loader, criterion_eval, device)

        metrics = compute_metrics(y_true, y_pred, y_prob)
        inference_ms, fps = measure_inference_speed(model, test_loader, device)
        metrics["Training_Time_sec"] = round(train_time, 2)
        metrics["Total_Params"] = total_params
        metrics["Trainable_Params"] = trainable_params
        metrics["Model_Size_MB"] = round(model_size_mb, 2)
        metrics["Inference_ms_per_image"] = inference_ms
        metrics["FPS"] = fps
        metrics["Accuracy_per_Million_Params"] = round(metrics["Accuracy"] / (trainable_params / 1_000_000), 4) if trainable_params > 0 else 0.0
        metrics["Model_Type"] = "Single"

        print(f"\n{model_name} Test Metrics:")
        print(metrics)

        trained_models[model_name] = model
        histories[model_name] = history
        single_results[model_name] = metrics
        single_outputs[model_name] = (y_true, y_pred)
        val_true_store[model_name] = val_true
        val_prob_store[model_name] = val_prob
        test_prob_store[model_name] = y_prob
        curves_store[model_name] = (y_true, y_prob)

    phase1_df = pd.DataFrame(single_results).T
    safe_save_csv(phase1_df, "phase1_single_deep_learning_results.csv")

    # ------------------------------------------------------------
    # PHASE 2 — ENSEMBLE LEARNING
    # ------------------------------------------------------------
    common_test_true = single_outputs["CustomCNN"][0]
    phase2_df, ensemble_outputs, ensemble_curves = build_ensemble_results(
        model_names=list(single_model_configs.keys()),
        test_true=common_test_true,
        test_prob_store=test_prob_store,
        val_true_store=val_true_store,
        val_prob_store=val_prob_store,
    )
    curves_store.update(ensemble_curves)
    save_ensemble_vs_single_summary(phase1_df, phase2_df)

    # ------------------------------------------------------------
    # PHASE 3 — OPTIMIZATION
    # ------------------------------------------------------------
    print_section("PHASE 3 — OPTIMIZATION")
    print("Main models already use augmentation, WeightedRandomSampler, early stopping, Adam, and ReduceLROnPlateau.")
    tuning_df = run_hyperparameter_tuning(device)
    ablation_df = run_ablation_study(test_loader, device)

    # ------------------------------------------------------------
    # PHASE 4 — EXPLAINABILITY AND ANALYSIS
    # ------------------------------------------------------------
    print_section("PHASE 4 — EXPLAINABILITY AND ANALYSIS")

    for model_name, model in trained_models.items():
        print(f"Extracting {model_name} embeddings for t-SNE...")
        train_feats, train_labels = extract_embeddings(model, train_loader, device)
        test_feats, test_labels = extract_embeddings(model, test_loader, device)
        feature_store[model_name] = {
            "train_features": train_feats,
            "train_labels": train_labels,
            "test_features": test_feats,
            "test_labels": test_labels,
        }
        plot_tsne(test_feats, test_labels, model_name)

    plot_gradcam(trained_models["CustomCNN"], "CustomCNN", trained_models["CustomCNN"].features[-3], test_loader, device)
    plot_gradcam(trained_models["MobileNetV2"], "MobileNetV2", trained_models["MobileNetV2"].features[-1], test_loader, device)
    plot_gradcam(trained_models["ResNet18"], "ResNet18", trained_models["ResNet18"].backbone[-2][-1].conv2, test_loader, device)
    if "EfficientNetB0" in trained_models:
        plot_gradcam(trained_models["EfficientNetB0"], "EfficientNetB0", trained_models["EfficientNetB0"].features[-1], test_loader, device)

    final_results = pd.concat([phase1_df, phase2_df], axis=0, sort=False)
    safe_save_csv(final_results, "final_single_and_ensemble_results.csv")

    plot_training_curves(histories)
    plot_metric_bar(final_results, "viz_single_vs_ensemble_performance.png")
    plot_efficiency(phase1_df)
    plot_macro_roc(curves_store)
    plot_macro_pr(curves_store)

    # Confusion matrices for all single models and ensembles
    for model_name, (y_true, y_pred) in single_outputs.items():
        plot_confusion_matrix(y_true, y_pred, f"Confusion Matrix — {model_name}", f"viz_cm_{model_name}.png")
    for model_name, (y_true, y_pred) in ensemble_outputs.items():
        safe_name = model_name.replace("+", "_").replace(" ", "_")
        plot_confusion_matrix(y_true, y_pred, f"Confusion Matrix — {model_name}", f"viz_cm_{safe_name}.png")

    # Per-class F1 heatmap for main models
    f1_report = {}
    report_outputs = {**single_outputs, **ensemble_outputs}
    for model_name, (y_true, y_pred) in report_outputs.items():
        report = classification_report(
            y_true,
            y_pred,
            target_names=[c.replace("im_", "") for c in CLASS_NAMES],
            output_dict=True,
            zero_division=0,
        )
        f1_report[model_name] = {
            class_name.replace("im_", ""): report[class_name.replace("im_", "")]["f1-score"]
            for class_name in CLASS_NAMES
        }

    f1_df = pd.DataFrame(f1_report).T
    safe_save_csv(f1_df, "per_class_f1_scores_single_and_ensemble.csv")
    plt.figure(figsize=(13, 6))
    sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Per-class F1-score Heatmap: Single Models vs Ensembles")
    plt.xlabel("Class")
    plt.ylabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    safe_save_fig("viz_per_class_f1_heatmap_single_ensemble.png", dpi=150)
    plt.close()

    # Best model summary: best single, best ensemble, best lightweight
    save_best_model_summary(final_results)

    print_section("EXPERIMENT COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("\nFinal thesis structure implemented:")
    print("Phase 1 — Single Deep Learning Models: CustomCNN, MobileNetV2, ResNet18")
    print("Phase 2 — Ensemble Learning: Soft Voting, Weighted Voting, Stacking")
    print("Phase 3 — Optimization: hyperparameter tuning, augmentation, sampler, early stopping")
    print("Phase 4 — Explainability & Analysis: Grad-CAM, confusion matrix, ROC, PR curve, ablation study")


if __name__ == "__main__":
    main()
