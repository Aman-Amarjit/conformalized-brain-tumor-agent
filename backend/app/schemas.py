from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ConformalPredictResponse(BaseModel):
    prediction_set: List[str] = Field(..., description="Calibrated prediction set of diagnosis labels")
    is_confident: bool = Field(..., description="True if set size is exactly 1, False if ambiguous or empty")
    set_size: int = Field(..., description="Size of the prediction set")
    target_coverage: float = Field(..., description="Statistical coverage guarantee level (1 - alpha)")
    alpha: float = Field(..., description="Significance level alpha")
    quantile_applied: float = Field(..., description="Conformal threshold q_hat applied")
    softmax_probabilities: Dict[str, float] = Field(..., description="Raw CNN softmax probabilities")
    abstention_triage_flag: bool = Field(..., description="True if flagged for human radiologist review")
    triage_message: str = Field(..., description="Clinical triage advice statement")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
    sample_id: Optional[str] = None

class CuratedSampleItem(BaseModel):
    id: str
    filename: str
    type: str
    label: str
    true_class: str
    expected_set: List[str]
    image_url: str

class ConformalMetricsResponse(BaseModel):
    num_train: int
    num_calibration: int
    num_test: int
    default_alpha: float
    default_target_coverage: float
    classes: List[str]
    quantiles: Dict[str, float]
    metrics: Dict[str, Any]

class AuditTrailItem(BaseModel):
    timestamp: str
    sample_name: str
    prediction_set: List[str]
    is_confident: bool
    target_coverage: float
    abstention_triage_flag: bool
    triage_message: str
