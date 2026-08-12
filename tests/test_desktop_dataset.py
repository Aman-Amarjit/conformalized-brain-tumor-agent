import os
import json
import time
from PIL import Image
import numpy as np
import torch
from backend.app.conformal_service import engine

DESKTOP_DATASET_DIR = '/home/aman-amarjit/Desktop/brain tumour detection/archive'

def test_desktop_scans():
    print("=========================================================================")
    print("🧪 EVALUATING USER DESKTOP DATASET SCANS (/home/aman-amarjit/Desktop/...)")
    print("=========================================================================")

    if not os.path.exists(DESKTOP_DATASET_DIR):
        print(f"❌ Desktop dataset directory not found at: {DESKTOP_DATASET_DIR}")
        return

    # Reload engine to ensure newly trained weights are loaded
    engine.load_engine()

    folders = [
        ('no', 'Normal Scan'),
        ('yes', 'Glioma')
    ]

    total_scans = 0
    correct_scans = 0
    in_set_scans = 0

    for folder_name, true_class in folders:
        folder_path = os.path.join(DESKTOP_DATASET_DIR, folder_name)
        if not os.path.exists(folder_path):
            continue
        
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:50]
        
        cls_correct = 0
        cls_total = 0

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                img_pil = Image.open(fpath)
                res = engine.predict_conformal(img_pil, alpha=0.05)
                
                prediction_set = res['prediction_set']
                sorted_probs = sorted(res['softmax_probabilities'].items(), key=lambda x: x[1], reverse=True)
                top_class = sorted_probs[0][0]

                cls_total += 1
                total_scans += 1

                if true_class in prediction_set:
                    in_set_scans += 1

                if top_class == true_class:
                    cls_correct += 1
                    correct_scans += 1
                else:
                    if fname in ['3 no.jpg', '1 no.jpeg', '10 no.jpg']:
                        print(f"   ℹ️ {fname} ({true_class}): Top Pred = {top_class} ({sorted_probs[0][1]*100:.1f}%), Probabilities = {res['softmax_probabilities']}")

            except Exception as e:
                print(f"Error testing {fpath}: {e}")

        acc = round((cls_correct / cls_total) * 100, 1) if cls_total > 0 else 0
        print(f"   ✓ Folder [{folder_name}] (True: {true_class}): {acc}% Accuracy ({cls_correct}/{cls_total})")

    overall_acc = round((correct_scans / total_scans) * 100, 1) if total_scans > 0 else 0
    overall_cov = round((in_set_scans / total_scans) * 100, 1) if total_scans > 0 else 0

    print("-" * 75)
    print(f"📊 DESKTOP SCANS OVERALL ACCURACY: {overall_acc}% ({correct_scans}/{total_scans})")
    print(f"🛡️ DESKTOP SCANS CONFORMAL COVERAGE: {overall_cov}% ({in_set_scans}/{total_scans})")
    print("=========================================================================\n")

if __name__ == '__main__':
    test_desktop_scans()
