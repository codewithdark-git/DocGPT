import base64
import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse, DiseaseType
from app.services.langchain_agents import DiseaseAgent

router = APIRouter()

@router.post("/", response_model=PredictionResponse)
async def predict_disease(request: PredictionRequest) -> PredictionResponse:
    """
    Predict disease from an uploaded image.
    
    Parameters:
    - disease_type: Type of disease to predict
    - image_base64: Base64 encoded image data
    
    Returns:
    - Prediction results including confidence score and analysis
    """
    try:
        # Decode base64 image and save temporarily
        image_data = base64.b64decode(request.image_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name

        # Initialize disease agent and get prediction
        agent = DiseaseAgent(img_path=temp_path, task=request.disease_type.value)
        result = agent.response()

        # Clean up temporary file
        Path(temp_path).unlink()

        if "error" in result:
            return PredictionResponse(
                disease_type=request.disease_type,
                prediction="",
                confidence=0.0,
                analysis="",
                status="error",
                error_message=str(result["error"])
            )

        return PredictionResponse(
            disease_type=request.disease_type,
            prediction=result["prediction"],
            confidence=result["confidence"],
            analysis=result["analysis"],
            status="success"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing prediction: {str(e)}"
        )