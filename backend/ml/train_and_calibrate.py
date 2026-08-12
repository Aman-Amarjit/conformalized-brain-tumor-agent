import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

MODEL_PATH = 'backend/data/brain_tumor_model.pth'
CALIB_PATH = 'backend/data/calibration_metadata.json'
DATASET_DIR = '/home/aman-amarjit/.cache/kagglehub/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/2'
DESKTOP_DATASET_DIR = '/home/aman-amarjit/Desktop/brain tumour detection/archive'

CLASSES = ["Glioma", "Meningioma", "Pituitary Tumor", "Normal Scan"]
CLASS_MAP = {
    'glioma': 0,
    'meningioma': 1,
    'pituitary': 2,
    'notumor': 3
}
NUM_CLASSES = len(CLASSES)

transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_eval = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def crop_to_brain_bounding_box(image_pil: Image.Image) -> Image.Image:
    gray = image_pil.convert('L')
    img_np = np.array(gray)
    h, w = img_np.shape
    mask = (img_np > 20) & (img_np < 248)
    if not np.any(mask):
        return image_pil
    row_counts = np.sum(mask, axis=1)
    col_counts = np.sum(mask, axis=0)
    valid_rows = np.where(row_counts > (0.05 * w))[0]
    valid_cols = np.where(col_counts > (0.05 * h))[0]
    if len(valid_rows) > 0 and len(valid_cols) > 0:
        min_y, max_y = int(valid_rows[0]), int(valid_rows[-1])
        min_x, max_x = int(valid_cols[0]), int(valid_cols[-1])
    else:
        y_indices, x_indices = np.where(mask)
        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
    w_box = max_x - min_x
    h_box = max_y - min_y
    pad_x = int(0.03 * w_box)
    pad_y = int(0.03 * h_box)
    crop_min_x = max(0, min_x - pad_x)
    crop_max_x = min(w, max_x + pad_x)
    crop_min_y = max(0, min_y - pad_y)
    crop_max_y = min(h, max_y + pad_y)
    return image_pil.crop((crop_min_x, crop_min_y, crop_max_x, crop_max_y))

def load_and_preprocess_dataset():
    print(f"Loading primary Kaggle dataset from: {DATASET_DIR}")
    images = []
    labels = []
    filepaths = []

    # 1. Load Kaggle 7,200 dataset
    splits = ['Training', 'Testing']
    for split in splits:
        split_dir = os.path.join(DATASET_DIR, split)
        if not os.path.exists(split_dir):
            continue
        for folder_name, class_idx in CLASS_MAP.items():
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.exists(folder_path):
                continue
            files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for fname in files:
                fpath = os.path.join(folder_path, fname)
                try:
                    img = Image.open(fpath).convert('RGB')
                    cropped = crop_to_brain_bounding_box(img)
                    tensor_img = transform_eval(cropped)
                    images.append(tensor_img)
                    labels.append(class_idx)
                    filepaths.append((fpath, class_idx))
                except Exception:
                    continue

    # 2. Load User Desktop dataset (/home/aman-amarjit/Desktop/brain tumour detection/archive)
    if os.path.exists(DESKTOP_DATASET_DIR):
        print(f"Loading user Desktop dataset from: {DESKTOP_DATASET_DIR}")
        desktop_folders = [
            (os.path.join(DESKTOP_DATASET_DIR, 'no'), 3),
            (os.path.join(DESKTOP_DATASET_DIR, 'yes'), 0),
            (os.path.join(DESKTOP_DATASET_DIR, 'brain_tumor_dataset', 'no'), 3),
            (os.path.join(DESKTOP_DATASET_DIR, 'brain_tumor_dataset', 'yes'), 0)
        ]
        for folder_path, class_idx in desktop_folders:
            if not os.path.exists(folder_path):
                continue
            files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for fname in files:
                fpath = os.path.join(folder_path, fname)
                try:
                    img = Image.open(fpath).convert('RGB')
                    cropped = crop_to_brain_bounding_box(img)
                    tensor_img = transform_eval(cropped)
                    images.append(tensor_img)
                    labels.append(class_idx)
                    filepaths.append((fpath, class_idx))
                except Exception:
                    continue

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    print(f"Successfully loaded {len(images_tensor)} total real MRI brain scans across {NUM_CLASSES} classes.")
    return images_tensor, labels_tensor, filepaths

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_and_calibrate():
    X, y, filepaths = load_and_preprocess_dataset()
    num_samples = len(X)

    indices = list(range(num_samples))
    random.shuffle(indices)

    n_train = int(0.70 * num_samples)
    n_cal = int(0.15 * num_samples)
    
    train_idx = indices[:n_train]
    cal_idx = indices[n_train:n_train + n_cal]
    test_idx = indices[n_train + n_cal:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_cal, y_cal = X[cal_idx], y[cal_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Splits: Train = {len(X_train)} | Calibration = {len(X_cal)} | Test = {len(X_test)}")

    model = BrainTumorCNN(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    batch_size = 64
    epochs = 15

    print("Training PyTorch CNN Backbone on Combined Dataset...")
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        correct = 0

        for i in range(0, X_train.size(0), batch_size):
            indices_b = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices_b], y_train[indices_b]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()

        acc = correct / len(X_train)
        avg_loss = epoch_loss / len(X_train)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Loss: {avg_loss:.4f} - Train Accuracy: {acc*100:.2f}%")

    model.eval()

    # 3. Conformal Calibration on Calibration Split
    print("Computing non-conformity calibration scores on calibration split...")
    with torch.no_grad():
        cal_logits = model(X_cal)
        cal_probs = torch.softmax(cal_logits, dim=1).numpy()

    cal_scores = []
    for i in range(len(y_cal)):
        true_label = y_cal[i].item()
        true_prob = cal_probs[i, true_label]
        s_i = 1.0 - true_prob
        cal_scores.append(s_i)

    cal_scores = np.sort(cal_scores)
    n_cal_samples = len(cal_scores)

    alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    quantiles_dict = {}

    print("Evaluating empirical coverage on holdout test set...")
    with torch.no_grad():
        test_logits = model(X_test)
        test_probs = torch.softmax(test_logits, dim=1).numpy()

    for alpha in alphas:
        q_level = np.ceil((n_cal_samples + 1) * (1 - alpha)) / n_cal_samples
        q_level = min(1.0, max(0.0, q_level))
        q_hat = float(np.quantile(cal_scores, q_level, method='higher'))
        quantiles_dict[f"{alpha:.2f}"] = round(q_hat, 4)

        covered = 0
        set_sizes = []
        for i in range(len(y_test)):
            true_label = y_test[i].item()
            pred_set = [c for c in range(NUM_CLASSES) if (1.0 - test_probs[i, c]) <= q_hat]
            if true_label in pred_set:
                covered += 1
            set_sizes.append(len(pred_set))

        emp_cov = covered / len(y_test)
        avg_set_size = np.mean(set_sizes)
        print(f"Alpha {alpha:.2f} (Target {int((1-alpha)*100)}%) -> Empirical Coverage: {emp_cov*100:.1f}% | Avg Set Size: {avg_set_size:.2f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved trained PyTorch model weights to: {MODEL_PATH}")

    calib_metadata = {
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "class_map": CLASS_MAP,
        "n_train": len(X_train),
        "n_cal": len(X_cal),
        "n_test": len(X_test),
        "quantiles": quantiles_dict,
        "curated_samples": [
            {
                "id": "study_conf_1",
                "filename": "study_conf_1.jpg",
                "type": "Pituitary Microadenoma",
                "label": "Study #1042 — Confident Diagnostic State",
                "true_class": "Pituitary Tumor",
                "expected_set": ["Pituitary Tumor"],
                "image_url": "/api/curated-samples/study_conf_1.jpg"
            },
            {
                "id": "study_conf_2",
                "filename": "study_conf_2.jpg",
                "type": "Pituitary Macroadenoma",
                "label": "Study #1043 — Confident Diagnostic State",
                "true_class": "Pituitary Tumor",
                "expected_set": ["Pituitary Tumor"],
                "image_url": "/api/curated-samples/study_conf_2.jpg"
            },
            {
                "id": "study_ambig_1",
                "filename": "study_ambig_1.jpg",
                "type": "Low-Grade Glioma",
                "label": "Study #1044 — Ambiguous State (Triage Required)",
                "true_class": "Glioma",
                "expected_set": ["Glioma"],
                "image_url": "/api/curated-samples/study_ambig_1.jpg"
            },
            {
                "id": "study_ambig_2",
                "filename": "study_ambig_2.jpg",
                "type": "High-Grade Glioma",
                "label": "Study #1045 — Ambiguous State (Triage Required)",
                "true_class": "Glioma",
                "expected_set": ["Glioma"],
                "image_url": "/api/curated-samples/study_ambig_2.jpg"
            }
        ]
    }

    with open(CALIB_PATH, 'w') as f:
        json.dump(calib_metadata, f, indent=2)

    print("Re-calibrated model on combined dataset!")

if __name__ == '__main__':
    train_and_calibrate()
