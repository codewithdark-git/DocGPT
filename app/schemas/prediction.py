from enum import Enum
from pydantic import BaseModel

class DiseaseType(str, Enum):
    SKIN_CANCER = "skin_cancer"
    PNEUMONIA = "pneumonia"
    BRAIN_TUMOR = "brain_tumor"
    EYE_DISEASE = "eye_disease"
    HEART_DISEASE = "heart_disease"

class PredictionResponse(BaseModel):
    disease_type: DiseaseType
    prediction: str
    confidence: float
    analysis: str
    status: str = "success"
    error_message: str | None = None 