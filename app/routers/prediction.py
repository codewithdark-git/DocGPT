from fastapi import APIRouter, HTTPException, UploadFile, File
from app.schemas.prediction import PredictionResponse, DiseaseType
from app.services.langchain_agents import DiseaseAgent
from langchain.schema import AIMessage
import tempfile
import os
import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO)

@router.post("/", response_model=PredictionResponse)
async def predict_disease(
    disease_type: DiseaseType,
    file: UploadFile = File(...)
) -> PredictionResponse:
    temp_path = None
    try:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            contents = await file.read()
            temp_file.write(contents)

        # Get prediction from agent
        agent = DiseaseAgent(img_path=temp_path, task=disease_type.value)
        result = agent.response()
        
        # Debug log
        logging.info(f"Agent Response: {result}")
        logging.info(f"Response Type: {type(result)}")

        if not result or not isinstance(result, dict):
            raise HTTPException(status_code=500, detail=f"Invalid response from prediction service: {result}")

        # Extract and format the analysis
        analysis = result.get("analysis", "")
        prediction = result.get("prediction", "")
        confidence = result.get("confidence", 0.0)

        logging.info(f"Analysis: {analysis}, Type: {type(analysis)}")
        logging.info(f"Prediction: {prediction}, Type: {type(prediction)}")
        logging.info(f"Confidence: {confidence}, Type: {type(confidence)}")

        if isinstance(analysis, AIMessage):
            analysis = analysis.content
        elif not isinstance(analysis, str):
            analysis = str(analysis)

        # Create response
        response = PredictionResponse(
            disease_type=disease_type,
            prediction=str(prediction),
            confidence=float(confidence),
            analysis=analysis,
            status="success"
        )

        return response

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing prediction: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass