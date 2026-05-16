# ============================================================
# Experiment 3 — SIPaKMeD Image Classification Pipeline
# Simplified and safer version for FYP
#
# Phase 1: CNN and ResNet18 single-model baselines
# Phase 2: Feature extraction + t-SNE + Grad-CAM
# Phase 3: Untuned hybrids
#          1) CNN embeddings + XGBoost
#          2) CNN + ResNet18 equal soft voting
# Phase 4: Tuned hybrids
#          1) Tuned CNN + XGBoost
#          2) AUC-weighted CNN + ResNet18
#
# Removed:
#   - stacking ensemble
#   - second fine-tuning of CNN/ResNet18
#   - weighted loss, because WeightedRandomSampler is already used
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms, models

from PIL import Image

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.manifold import TSNE
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = Path(r"D:\vscode\Code\FYP\outputs\exp3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = r"E:\FYP"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DATA_ROOT = Path(r"E:\sipakmed")

CLASS_NAMES = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate",
]

NUM_CLASSES = len(CLASS_NAMES)

BINARY_MAP = {
    "im_Dyskeratotic": 1,
    "im_Koilocytotic": 1,
    "im_Metaplastic": 0,
    "im_Parabasal": 0,
    "im_Superficial-Intermediate": 0,
}

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR_CNN = 1e-3
LR_RESNET = 1e-4
WEIGHT_DECAY = 1e-4


def out(fname):
    return OUTPUT_DIR / fname


def safe_save_fig(fname, **kwargs):
    plt.savefig(out(fname), bbox_inches="tight", **kwargs)


def safe_save_csv(df, fname):
    path = out(fname)
    df.to_csv(path, index=True)
    return path


def print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ============================================================
# 2. DATASET AND TRANSFORMS
# ============================================================

def build_transforms(phase="train"):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
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
            class_dir = self.data_root / class_name / class_name

            if not class_dir.exists():
                class_dir = self.data_root / class_name

            if not class_dir.exists():
                print(f"[WARN] Missing folder: {class_dir}")
                continue

            image_files = list(class_dir.glob("*.bmp"))
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
        self.indices = indices
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


def load_datasets(data_root):
    print_section("DATA LOADING AND PREPROCESSING")

    full_dataset = SIPaKMeDDataset(data_root, transform=None)

    if len(full_dataset) == 0:
        raise RuntimeError(
            f"No images found under {data_root}. Please check DATA_ROOT."
        )

    labels = [label for _, label in full_dataset.samples]
    indices = list(range(len(full_dataset)))

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp
    )

    print(f"Train size: {len(train_idx)}")
    print(f"Validation size: {len(val_idx)}")
    print(f"Test size: {len(test_idx)}")

    train_dataset = TransformSubset(full_dataset, train_idx, build_transforms("train"))
    val_dataset = TransformSubset(full_dataset, val_idx, build_transforms("val"))
    test_dataset = TransformSubset(full_dataset, test_idx, build_transforms("test"))

    train_labels = [labels[i] for i in train_idx]
    train_counts = Counter(train_labels)

    sample_weights = [
        1.0 / train_counts[labels[i]]
        for i in train_idx
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return full_dataset, train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


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
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        x = self.features(x)
        x = self.gap(x)
        return x.squeeze(-1).squeeze(-1)


class ResNet18WithFeatures(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
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
            base.avgpool
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)

    def extract_features(self, x):
        x = self.backbone(x)
        return x.squeeze(-1).squeeze(-1)


# ============================================================
# 4. TRAINING AND EVALUATION
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

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


def evaluate_model(model, loader, criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_true = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

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

    all_probs = np.vstack(all_probs)

    return total_loss / total, correct / total, np.array(all_true), np.array(all_preds), all_probs


def compute_metrics(y_true, y_pred, y_prob):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    try:
        auc = roc_auc_score(
            y_true_bin,
            y_prob,
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_true_bin,
            y_prob,
            average="macro"
        )
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


def train_model(
    model,
    train_loader,
    val_loader,
    model_name,
    lr,
    num_epochs=NUM_EPOCHS,
    patience=6
):
    model = model.to(device)

    # Important:
    # We use normal CrossEntropyLoss here.
    # Class imbalance is handled by WeightedRandomSampler.
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    print(f"\nTraining {model_name}...")

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_acc, _, _, _ = evaluate_model(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} | "
            f"Time={time.time() - start:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
            torch.save(model.state_dict(),os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
            )
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def extract_embeddings(model, loader):
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.extract_features(images)

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.vstack(all_features), np.array(all_labels)


# ============================================================
# 5. GRAD-CAM
# ============================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
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


# ============================================================
# 6. VISUALISATION HELPERS
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
    safe_save_fig("viz1_validation_accuracy_curves.png", dpi=150)
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
        yticklabels=[c.replace("im_", "") for c in CLASS_NAMES]
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    safe_save_fig(filename, dpi=150)
    plt.close()


def plot_metric_bar(results_df, filename):
    metrics = ["Accuracy", "Recall", "F1", "AUC", "PR-AUC"]

    results_df[metrics].plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    safe_save_fig(filename, dpi=150)
    plt.close()


def plot_tsne(features, labels, model_name):
    n_samples = min(len(features), 1000)
    sample_idx = np.random.choice(len(features), n_samples, replace=False)

    features_sample = features[sample_idx]
    labels_sample = labels[sample_idx]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_sample)

    tsne = TSNE(
        n_components=2,
        random_state=SEED,
        perplexity=30,
        learning_rate="auto",
        init="pca"
    )

    features_2d = tsne.fit_transform(features_scaled)

    plt.figure(figsize=(9, 7))

    for class_idx in range(NUM_CLASSES):
        mask = labels_sample == class_idx
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            s=25,
            alpha=0.7,
            label=CLASS_NAMES[class_idx].replace("im_", "")
        )

    plt.title(
        f"t-SNE of {model_name} Embeddings\n"
        "This visualisation suggests possible class-separation patterns."
    )
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    safe_save_fig(f"viz_tsne_{model_name}.png", dpi=150)
    plt.close()


def plot_gradcam(model, model_name, target_layer, test_loader):
    gradcam = GradCAM(model, target_layer)

    images, labels = next(iter(test_loader))
    images = images[:3]
    labels = labels[:3]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for i in range(3):
        img_tensor = images[i]
        true_label = labels[i].item()

        img_np = img_tensor.numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np * std + mean, 0, 1)

        cam, pred_class = gradcam.generate(img_tensor)

        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize((IMG_SIZE, IMG_SIZE))
        cam_resized = np.array(cam_img) / 255.0

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(
            f"True: {CLASS_NAMES[true_label].replace('im_', '')}\n"
            f"Pred: {CLASS_NAMES[pred_class].replace('im_', '')}"
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(img_np)
        axes[1, i].imshow(cam_resized, cmap="jet", alpha=0.45)
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis("off")

    plt.suptitle(f"Grad-CAM Heatmaps — {model_name}")
    plt.tight_layout()
    safe_save_fig(f"viz_gradcam_{model_name}.png", dpi=150)
    plt.close()


def compute_binary_metrics(y_true_5, y_pred_5, y_prob_5):
    binary_label_map = np.array([
        BINARY_MAP[class_name]
        for class_name in CLASS_NAMES
    ])

    y_true_bin = binary_label_map[y_true_5]
    y_pred_bin = binary_label_map[y_pred_5]

    # Abnormal probability = Dyskeratotic + Koilocytotic
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


# ============================================================
def main():
    # 7. MAIN PIPELINE
    # ============================================================

    all_results = {}
    all_histories = {}

    full_dataset, train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = load_datasets(DATA_ROOT)

    criterion_eval = nn.CrossEntropyLoss()


    # ============================================================
    # PHASE 1 — SINGLE MODEL BASELINES
    # ============================================================

    print_section("PHASE 1 — SINGLE-MODEL BASELINES")

    cnn_model = CustomCNN(num_classes=NUM_CLASSES, dropout_p=0.5)

    def replace_relu(model):
        for name, module in model.named_children():
            if isinstance(module, nn.ReLU):
                setattr(model, name, nn.ReLU(inplace=False))
            else:
                replace_relu(module)

    replace_relu(cnn_model)

    cnn_model, cnn_history = train_model(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="p1_cnn",
        lr=LR_CNN,
        num_epochs=NUM_EPOCHS,
        patience=6
    )

    cnn_loss, cnn_acc, cnn_true, cnn_preds, cnn_probs = evaluate_model(
        cnn_model,
        test_loader,
        criterion_eval
    )

    cnn_metrics = compute_metrics(cnn_true, cnn_preds, cnn_probs)

    print("\nCNN Metrics:")
    print(cnn_metrics)

    resnet_model = ResNet18WithFeatures(num_classes=NUM_CLASSES, pretrained=True)
    replace_relu(resnet_model)

    resnet_model, resnet_history = train_model(
        model=resnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="p1_resnet18",
        lr=LR_RESNET,
        num_epochs=NUM_EPOCHS,
        patience=6
    )

    res_loss, res_acc, res_true, res_preds, res_probs = evaluate_model(
        resnet_model,
        test_loader,
        criterion_eval
    )

    resnet_metrics = compute_metrics(res_true, res_preds, res_probs)

    print("\nResNet18 Metrics:")
    print(resnet_metrics)

    phase1_results = pd.DataFrame({
        "CNN": cnn_metrics,
        "ResNet18": resnet_metrics
    }).T

    all_results["Phase1_Baseline"] = phase1_results
    all_histories["CNN"] = cnn_history
    all_histories["ResNet18"] = resnet_history

    print("\nPhase 1 Summary:")
    print(phase1_results)

    safe_save_csv(phase1_results, "phase1_results.csv")


    # ============================================================
    # PHASE 2 — FEATURE EXTRACTION AND VISUALISATION
    # ============================================================

    print_section("PHASE 2 — FEATURE EXTRACTION AND VISUALISATION")

    print("Extracting CNN embeddings...")
    cnn_feats_train, cnn_labels_train = extract_embeddings(cnn_model, train_loader)
    cnn_feats_test, cnn_labels_test = extract_embeddings(cnn_model, test_loader)

    print("Extracting ResNet18 embeddings...")
    resnet_feats_train, resnet_labels_train = extract_embeddings(resnet_model, train_loader)
    resnet_feats_test, resnet_labels_test = extract_embeddings(resnet_model, test_loader)

    print(f"CNN embedding shape: {cnn_feats_train.shape}")
    print(f"ResNet18 embedding shape: {resnet_feats_train.shape}")

    plot_tsne(cnn_feats_test, cnn_labels_test, "CNN")
    plot_tsne(resnet_feats_test, resnet_labels_test, "ResNet18")

    plot_gradcam(
        model=cnn_model,
        model_name="CNN",
        target_layer=cnn_model.features[-3],
        test_loader=test_loader
    )

    plot_gradcam(
        model=resnet_model,
        model_name="ResNet18",
        target_layer=resnet_model.backbone[-2][-1].conv2,
        test_loader=test_loader
    )

    # CNN embedding activation importance
    mean_abs_activation = np.abs(cnn_feats_test).mean(axis=0)
    top_idx = np.argsort(mean_abs_activation)[::-1][:20]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(top_idx)), mean_abs_activation[top_idx])
    plt.xticks(range(len(top_idx)), [f"F{i}" for i in top_idx], rotation=45)
    plt.title("Top 20 CNN Embedding Dimensions by Mean Absolute Activation")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Mean |Activation|")
    plt.grid(axis="y", alpha=0.3)
    safe_save_fig("viz_cnn_top20_embedding_activation.png", dpi=150)
    plt.close()


    # ============================================================
    # PHASE 3 — UNTUNED HYBRID MODELS
    # ============================================================

    print_section("PHASE 3 — UNTUNED HYBRID MODELS")

    print("\nHybrid 1: CNN embeddings + XGBoost")

    scaler_cnn = StandardScaler()
    X_train_cnn = scaler_cnn.fit_transform(cnn_feats_train)
    X_test_cnn = scaler_cnn.transform(cnn_feats_test)

    train_counts = Counter(cnn_labels_train)
    n_train = len(cnn_labels_train)

    xgb_sample_weights = np.array([
        n_train / (NUM_CLASSES * train_counts[label])
        for label in cnn_labels_train
    ])

    xgb_p3 = XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist"
    )

    xgb_p3.fit(
        X_train_cnn,
        cnn_labels_train,
        sample_weight=xgb_sample_weights
    )

    cnn_xgb_probs = xgb_p3.predict_proba(X_test_cnn)
    cnn_xgb_preds = cnn_xgb_probs.argmax(axis=1)

    cnn_xgb_metrics = compute_metrics(
        cnn_labels_test,
        cnn_xgb_preds,
        cnn_xgb_probs
    )

    print("CNN + XGBoost Metrics:")
    print(cnn_xgb_metrics)

    print("\nHybrid 2: CNN + ResNet18 equal soft voting")

    equal_vote_probs = 0.5 * cnn_probs + 0.5 * res_probs
    equal_vote_preds = equal_vote_probs.argmax(axis=1)

    equal_vote_metrics = compute_metrics(
        cnn_true,
        equal_vote_preds,
        equal_vote_probs
    )

    print("CNN + ResNet18 Equal Soft Voting Metrics:")
    print(equal_vote_metrics)

    phase3_results = pd.DataFrame({
        "CNN": cnn_metrics,
        "ResNet18": resnet_metrics,
        "CNN+XGBoost": cnn_xgb_metrics,
        "CNN+ResNet18_EqualVote": equal_vote_metrics
    }).T

    all_results["Phase3_Hybrid"] = phase3_results

    print("\nPhase 3 Summary:")
    print(phase3_results)

    safe_save_csv(phase3_results, "phase3_results.csv")


    # ============================================================
    # PHASE 4 — TUNED HYBRID MODELS
    # ============================================================

    print_section("PHASE 4 — TUNED HYBRID MODELS")

    print("\nTuning CNN + XGBoost...")

    xgb_param_grid = {
        "n_estimators": [100],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    xgb_grid = GridSearchCV(
        estimator=XGBClassifier(
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=1,
            tree_method="hist"
        ),
        param_grid=xgb_param_grid,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        n_jobs=1,
        verbose=1
    )

    xgb_grid.fit(
        X_train_cnn,
        cnn_labels_train,
        sample_weight=xgb_sample_weights
    )

    print(f"Best XGBoost parameters: {xgb_grid.best_params_}")
    print(f"Best CV macro F1: {xgb_grid.best_score_:.4f}")

    tuned_xgb = xgb_grid.best_estimator_

    tuned_cnn_xgb_probs = tuned_xgb.predict_proba(X_test_cnn)
    tuned_cnn_xgb_preds = tuned_cnn_xgb_probs.argmax(axis=1)

    tuned_cnn_xgb_metrics = compute_metrics(
        cnn_labels_test,
        tuned_cnn_xgb_preds,
        tuned_cnn_xgb_probs
    )

    print("\nTuned CNN + XGBoost Metrics:")
    print(tuned_cnn_xgb_metrics)


    print("\nComputing AUC-weighted CNN + ResNet18 soft voting...")

    _, _, val_true, _, cnn_val_probs = evaluate_model(
        cnn_model,
        val_loader,
        criterion_eval
    )

    _, _, _, _, res_val_probs = evaluate_model(
        resnet_model,
        val_loader,
        criterion_eval
    )

    val_true_bin = label_binarize(val_true, classes=list(range(NUM_CLASSES)))

    try:
        cnn_val_auc = roc_auc_score(
            val_true_bin,
            cnn_val_probs,
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        cnn_val_auc = 0.5

    try:
        res_val_auc = roc_auc_score(
            val_true_bin,
            res_val_probs,
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        res_val_auc = 0.5

    auc_sum = cnn_val_auc + res_val_auc

    if auc_sum == 0:
        w_cnn = 0.5
        w_resnet = 0.5
    else:
        w_cnn = cnn_val_auc / auc_sum
        w_resnet = res_val_auc / auc_sum

    print(f"CNN validation AUC: {cnn_val_auc:.4f}")
    print(f"ResNet18 validation AUC: {res_val_auc:.4f}")
    print(f"Final ensemble weights: CNN={w_cnn:.3f}, ResNet18={w_resnet:.3f}")

    auc_weighted_probs = w_cnn * cnn_probs + w_resnet * res_probs
    auc_weighted_preds = auc_weighted_probs.argmax(axis=1)

    auc_weighted_metrics = compute_metrics(
        cnn_true,
        auc_weighted_preds,
        auc_weighted_probs
    )

    print("\nAUC-weighted CNN + ResNet18 Metrics:")
    print(auc_weighted_metrics)

    phase4_results = pd.DataFrame({
        "Tuned_CNN+XGBoost": tuned_cnn_xgb_metrics,
        "AUCWeighted_CNN+ResNet18": auc_weighted_metrics
    }).T

    all_results["Phase4_Tuned"] = phase4_results

    print("\nPhase 4 Summary:")
    print(phase4_results)

    safe_save_csv(phase4_results, "phase4_results.csv")


    # ============================================================
    # 8. FINAL VISUALISATIONS AND REPORTS
    # ============================================================

    print_section("FINAL VISUALISATIONS AND REPORTS")

    final_results = pd.concat([
        phase1_results,
        phase3_results,
        phase4_results
    ])

    safe_save_csv(final_results, "final_all_results.csv")

    print("\nFinal All Results:")
    print(final_results)

    plot_training_curves(all_histories)
    plot_metric_bar(final_results, "viz_model_performance_comparison.png")

    plot_confusion_matrix(
        cnn_true,
        cnn_preds,
        "Confusion Matrix — CNN",
        "viz_cm_cnn.png"
    )

    plot_confusion_matrix(
        res_true,
        res_preds,
        "Confusion Matrix — ResNet18",
        "viz_cm_resnet18.png"
    )

    plot_confusion_matrix(
        cnn_labels_test,
        cnn_xgb_preds,
        "Confusion Matrix — CNN + XGBoost",
        "viz_cm_cnn_xgboost.png"
    )

    plot_confusion_matrix(
        cnn_true,
        equal_vote_preds,
        "Confusion Matrix — CNN + ResNet18 Equal Soft Voting",
        "viz_cm_equal_soft_voting.png"
    )

    plot_confusion_matrix(
        cnn_labels_test,
        tuned_cnn_xgb_preds,
        "Confusion Matrix — Tuned CNN + XGBoost",
        "viz_cm_tuned_cnn_xgboost.png"
    )

    plot_confusion_matrix(
        cnn_true,
        auc_weighted_preds,
        "Confusion Matrix — AUC-weighted CNN + ResNet18",
        "viz_cm_auc_weighted_ensemble.png"
    )


    # ROC curves, macro one-vs-rest
    plt.figure(figsize=(9, 7))

    roc_models = {
        "CNN": (cnn_true, cnn_probs),
        "ResNet18": (res_true, res_probs),
        "CNN+XGBoost": (cnn_labels_test, cnn_xgb_probs),
        "EqualVote_CNN+ResNet18": (cnn_true, equal_vote_probs),
        "Tuned_CNN+XGBoost": (cnn_labels_test, tuned_cnn_xgb_probs),
        "AUCWeighted_CNN+ResNet18": (cnn_true, auc_weighted_probs),
    }

    for model_name, (y_true, y_prob) in roc_models.items():
        y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

        try:
            fpr = {}
            tpr = {}

            for i in range(NUM_CLASSES):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])

            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
            mean_tpr = np.zeros_like(all_fpr)

            for i in range(NUM_CLASSES):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            mean_tpr /= NUM_CLASSES

            auc_score = roc_auc_score(
                y_true_bin,
                y_prob,
                multi_class="ovr",
                average="macro"
            )

            plt.plot(
                all_fpr,
                mean_tpr,
                linewidth=2,
                label=f"{model_name} AUC={auc_score:.3f}"
            )

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


    # Classification report heatmap for final models
    report_models = {
        "CNN": (cnn_true, cnn_preds),
        "ResNet18": (res_true, res_preds),
        "Tuned_CNN+XGBoost": (cnn_labels_test, tuned_cnn_xgb_preds),
        "AUCWeighted_CNN+ResNet18": (cnn_true, auc_weighted_preds),
    }

    f1_report = {}

    for model_name, (y_true, y_pred) in report_models.items():
        report = classification_report(
            y_true,
            y_pred,
            target_names=[c.replace("im_", "") for c in CLASS_NAMES],
            output_dict=True,
            zero_division=0
        )

        f1_report[model_name] = {
            class_name.replace("im_", ""): report[class_name.replace("im_", "")]["f1-score"]
            for class_name in CLASS_NAMES
        }

    f1_df = pd.DataFrame(f1_report).T

    plt.figure(figsize=(11, 5))
    sns.heatmap(
        f1_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu"
    )
    plt.title("Per-class F1-score Heatmap")
    plt.xlabel("Class")
    plt.ylabel("Model")
    plt.xticks(rotation=45, ha="right")
    safe_save_fig("viz_per_class_f1_heatmap.png", dpi=150)
    plt.close()

    safe_save_csv(f1_df, "per_class_f1_scores.csv")


    # Binary collapse report
    binary_results = {}

    binary_models = {
        "CNN": (cnn_true, cnn_preds, cnn_probs),
        "ResNet18": (res_true, res_preds, res_probs),
        "CNN+XGBoost": (cnn_labels_test, cnn_xgb_preds, cnn_xgb_probs),
        "EqualVote_CNN+ResNet18": (cnn_true, equal_vote_preds, equal_vote_probs),
        "Tuned_CNN+XGBoost": (cnn_labels_test, tuned_cnn_xgb_preds, tuned_cnn_xgb_probs),
        "AUCWeighted_CNN+ResNet18": (cnn_true, auc_weighted_preds, auc_weighted_probs),
    }

    for model_name, (y_true_5, y_pred_5, y_prob_5) in binary_models.items():
        binary_results[model_name] = compute_binary_metrics(
            y_true_5,
            y_pred_5,
            y_prob_5
        )

    binary_df = pd.DataFrame(binary_results).T

    print("\nBinary Normal vs Abnormal Results:")
    print(binary_df)

    safe_save_csv(binary_df, "binary_normal_vs_abnormal_results.csv")

    plt.figure(figsize=(10, 6))
    binary_df[["Binary_Accuracy", "Binary_Recall", "Binary_F1", "Binary_AUC"]].plot(
        kind="bar",
        figsize=(12, 6)
    )
    plt.title("Binary Classification Performance: Normal vs Abnormal")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    safe_save_fig("viz_binary_performance.png", dpi=150)
    plt.close()


    print_section("EXPERIMENT 3 COMPLETE")

    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("\nRecommended final models to compare in论文:")
    print("1. CNN baseline")
    print("2. ResNet18 baseline")
    print("3. Tuned CNN + XGBoost")
    print("4. AUC-weighted CNN + ResNet18")
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()
