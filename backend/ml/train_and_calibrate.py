import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import torchvision.transforms as transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET_DIR = '/home/aman-amarjit/.cache/kagglehub/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/2'
MODEL_SAVE_PATH = 'backend/data/brain_tumor_model.pth'
CALIB_SAVE_PATH = 'backend/data/calibration_metadata.json'
CURATED_DIR = 'backend/curated_samples'

os.makedirs('backend/data', exist_ok=True)
os.makedirs(CURATED_DIR, exist_ok=True)

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

def load_and_preprocess_dataset():
    print(f"Loading new 7,023-sample dataset from: {DATASET_DIR}")
    images = []
    labels = []
    filepaths = []

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
                    tensor_img = transform_eval(img)
                    images.append(tensor_img)
                    labels.append(class_idx)
                    filepaths.append((fpath, class_idx))
                except Exception:
                    continue

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    print(f"Successfully loaded {len(images_tensor)} real MRI brain scans across {NUM_CLASSES} classes.")
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    print("Training PyTorch CNN Backbone on new 7,023-sample dataset...")
    model.train()
    for epoch in range(15):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        scheduler.step()
        epoch_acc = (correct / total) * 100.0
        epoch_loss = running_loss / total
        print(f"Epoch {epoch+1:02d}/15 - Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_acc:.2f}%")

    model.eval()
    print("Computing non-conformity calibration scores on calibration split...")
    with torch.no_grad():
        cal_logits = model(X_cal)
        cal_probs = torch.softmax(cal_logits, dim=1).numpy()
        cal_y_true = y_cal.numpy()

    cal_scores = []
    for i in range(len(cal_probs)):
        true_class_prob = cal_probs[i, cal_y_true[i]]
        score = 1.0 - true_class_prob
        cal_scores.append(score)

    cal_scores = np.array(cal_scores)
    n_c = len(cal_scores)

    alpha_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    quantiles_dict = {}

    for alpha in alpha_levels:
        q_val = np.quantile(cal_scores, np.clip((1 - alpha) * (1 + 1 / n_c), 0.0, 1.0), method='higher')
        quantiles_dict[str(alpha)] = float(q_val)

    print("Evaluating empirical coverage on holdout test set...")
    with torch.no_grad():
        test_logits = model(X_test)
        test_probs = torch.softmax(test_logits, dim=1).numpy()
        test_y_true = y_test.numpy()

    test_coverage_results = {}
    for alpha in alpha_levels:
        q_hat = quantiles_dict[str(alpha)]
        covered = 0
        set_sizes = []
        for i in range(len(test_probs)):
            pred_set = [c for c in range(NUM_CLASSES) if (1.0 - test_probs[i, c]) <= q_hat]
            set_sizes.append(len(pred_set))
            if test_y_true[i] in pred_set:
                covered += 1
        emp_coverage = covered / len(test_probs)
        avg_set_size = np.mean(set_sizes)
        test_coverage_results[str(alpha)] = {
            "target_coverage": float(1 - alpha),
            "empirical_coverage": float(emp_coverage),
            "average_set_size": float(avg_set_size),
            "quantile_q_hat": quantiles_dict[str(alpha)]
        }
        print(f"Alpha {alpha:.2f} (Target {1-alpha:.0%}) -> Empirical Coverage: {emp_coverage*100:.1f}% | Avg Set Size: {avg_set_size:.2f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved trained PyTorch model weights to: {MODEL_SAVE_PATH}")

    # Curate Demo DICOM Studies from the test set
    curated_info = []
    q_95 = quantiles_dict["0.05"]
    
    confident_count = 0
    ambiguous_count = 0
    
    accession_ids_conf = ["ACC-2026-0891_AX_T2.dcm", "ACC-2026-0318_AX_T2.dcm"]
    accession_ids_ambig = ["ACC-2026-0447_FLAIR.dcm", "ACC-2026-1205_COR_T1.dcm"]

    for i in range(len(X_test)):
        orig_path, class_id = filepaths[test_idx[i]]
        probs = test_probs[i]
        pred_set = [CLASSES[c] for c in range(NUM_CLASSES) if (1.0 - probs[c]) <= q_95]
        
        is_conf = (len(pred_set) == 1)
        if is_conf and confident_count < 2:
            confident_count += 1
            acc_id = accession_ids_conf[confident_count - 1]
            dst_name = f"study_conf_{confident_count}.jpg"
            dst_path = os.path.join(CURATED_DIR, dst_name)
            img = Image.open(orig_path).convert('RGB')
            img.save(dst_path)
            curated_info.append({
                "id": f"study_conf_{confident_count}",
                "filename": dst_name,
                "type": "confident",
                "label": acc_id,
                "true_class": CLASSES[class_id],
                "expected_set": pred_set
            })
        elif not is_conf and ambiguous_count < 2:
            ambiguous_count += 1
            acc_id = accession_ids_ambig[ambiguous_count - 1]
            dst_name = f"study_ambig_{ambiguous_count}.jpg"
            dst_path = os.path.join(CURATED_DIR, dst_name)
            img = Image.open(orig_path).convert('RGB')
            img.save(dst_path)
            curated_info.append({
                "id": f"study_ambig_{ambiguous_count}",
                "filename": dst_name,
                "type": "ambiguous",
                "label": acc_id,
                "true_class": CLASSES[class_id],
                "expected_set": pred_set
            })

    metadata = {
        "classes": CLASSES,
        "num_classes": NUM_CLASSES,
        "n_train": len(X_train),
        "n_cal": len(X_cal),
        "n_test": len(X_test),
        "default_alpha": 0.05,
        "default_target_coverage": 0.95,
        "quantiles": quantiles_dict,
        "metrics": test_coverage_results,
        "curated_samples": curated_info
    }

    with open(CALIB_SAVE_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Re-calibrated model and exported DICOM accession studies!")

if __name__ == '__main__':
    train_and_calibrate()
