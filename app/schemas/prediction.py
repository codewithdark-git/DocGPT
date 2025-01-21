from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum

class DiseaseType(str, Enum):
    PNEUMONIA = "Pneumonia"
    MELANOMA = "Melanoma"
    BRAIN = "Brain"
    HEART = "Heart"

class PredictionRequest(BaseModel):
    disease_type: DiseaseType = Field(..., description="Type of disease to predict")
    image_base64: str = Field(..., description="Base64 encoded image data")

class PredictionResponse(BaseModel):
    disease_type: DiseaseType
    prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    analysis: str
    status: Literal["success", "error"] = "success"
    error_message: Optional[str] = None 