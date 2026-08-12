import os
import json
import time
from PIL import Image
import numpy as np
import torch
from backend.app.conformal_service import engine

DATASET_DIR = '/home/aman-amarjit/.cache/kagglehub/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/2/Testing'

CLASS_MAP = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'pituitary': 'Pituitary Tumor',
    'notumor': 'Normal Scan'
}

def test_on_real_dataset():
    print("=========================================================================")
    print("🧪 AUTOMATED EVALUATION ON REAL MRI CLINICAL DATASET (TEST SPLIT)")
    print("=========================================================================")

    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset directory not found at: {DATASET_DIR}")
        return

    results_by_class = {cls: {'total': 0, 'correct_single': 0, 'in_set': 0} for cls in CLASS_MAP.values()}
    total_scans = 0
    total_set_sizes = []
    start_time = time.time()

    for folder_name, true_class in CLASS_MAP.items():
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.exists(folder_path):
            continue
        
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:50] # 50 per class
        
        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                img_pil = Image.open(fpath)
                res = engine.predict_conformal(img_pil, alpha=0.05)
                
                prediction_set = res['prediction_set']
                top_pred = list(res['softmax_probabilities'].keys())[0] # highest prob
                
                results_by_class[true_class]['total'] += 1
                total_scans += 1
                total_set_sizes.append(res['set_size'])

                if true_class in prediction_set:
                    results_by_class[true_class]['in_set'] += 1
                
                # Check top single prediction
                sorted_probs = sorted(res['softmax_probabilities'].items(), key=lambda x: x[1], reverse=True)
                top_class = sorted_probs[0][0]
                if top_class == true_class:
                    results_by_class[true_class]['correct_single'] += 1
                    
            except Exception as e:
                print(f"Error testing {fpath}: {e}")

    elapsed = round(time.time() - start_time, 2)
    print(f"\n📊 EVALUATION COMPLETED IN {elapsed}s ON {total_scans} REAL CLINICAL SCAN SLICES\n")
    print(f"{'Class Name':<18} | {'Total Tested':<12} | {'Single Accuracy':<15} | {'Conformal Coverage (Target 95%)'}")
    print("-" * 75)

    overall_correct = 0
    overall_in_set = 0

    for cls_name, stats in results_by_class.items():
        tot = stats['total']
        if tot == 0:
            continue
        acc = round((stats['correct_single'] / tot) * 100, 1)
        cov = round((stats['in_set'] / tot) * 100, 1)
        overall_correct += stats['correct_single']
        overall_in_set += stats['in_set']
        print(f"{cls_name:<18} | {tot:<12} | {acc}% ({stats['correct_single']}/{tot}){'':<4} | {cov}% ({stats['in_set']}/{tot})")

    print("-" * 75)
    total_acc = round((overall_correct / total_scans) * 100, 1)
    total_cov = round((overall_in_set / total_scans) * 100, 1)
    avg_set_size = round(np.mean(total_set_sizes), 2)

    print(f"{'OVERALL TOTAL':<18} | {total_scans:<12} | {total_acc}% ({overall_correct}/{total_scans}){'':<4} | {total_cov}% Empirical Coverage")
    print(f"📈 Average Conformal Set Size (|C_0.05|): {avg_set_size}")
    print("=========================================================================\n")

if __name__ == '__main__':
    test_on_real_dataset()
