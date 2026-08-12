import os
import json
from PIL import Image
import numpy as np
import torch
from backend.app.conformal_service import engine

def test_model_loading():
    """Verify PyTorch model weights and calibration metadata load correctly"""
    assert engine.model is not None, "PyTorch model backbone failed to load"
    assert os.path.exists('backend/data/calibration_metadata.json'), "Calibration metadata missing"
    with open('backend/data/calibration_metadata.json') as f:
        meta = json.load(f)
    assert meta["num_classes"] == 4
    assert "0.05" in meta["quantiles"]

def test_annotation_resistant_cropping():
    """Verify cropping engine strips padding and handles arbitrary images"""
    # Create synthetic test scan with white text annotation on right edge
    test_img = Image.new('RGB', (400, 300), (0, 0, 0))
    np_img = np.array(test_img)
    # Add circular skull mask in center
    cy, cx = 150, 200
    y, x = np.ogrid[:300, :400]
    mask = (x - cx)**2 + (y - cy)**2 <= 80**2
    np_img[mask] = [120, 120, 120]
    # Add text annotation "1" on right edge
    np_img[10:40, 380:390] = [255, 255, 255]

    annotated_pil = Image.fromarray(np_img)
    cropped = engine.crop_to_brain_bounding_box(annotated_pil)
    
    # Bounding box should isolate central brain and exclude right edge annotation
    w, h = cropped.size
    assert w < 300, f"Crop failed to exclude right annotation margin, width was {w}"

def test_conformal_prediction_schema():
    """Verify prediction output schema contains all required conformal fields"""
    test_pil = Image.new('RGB', (200, 200), (50, 50, 50))
    res = engine.predict_conformal(test_pil, alpha=0.05)
    
    required_keys = [
        'prediction_set', 'is_confident', 'set_size', 'target_coverage',
        'alpha', 'quantile_applied', 'softmax_probabilities',
        'abstention_triage_flag', 'triage_message', 'inference_time_ms'
    ]
    for k in required_keys:
        assert k in res, f"Missing key in prediction response: {k}"
    
    assert res["target_coverage"] == 0.95
    assert res["alpha"] == 0.05

if __name__ == '__main__':
    print("Running System Verification Tests...")
    test_model_loading()
    print("✓ Model loading test passed.")
    test_annotation_resistant_cropping()
    print("✓ Annotation-resistant cropping test passed.")
    test_conformal_prediction_schema()
    print("✓ Conformal prediction response schema test passed.")
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
