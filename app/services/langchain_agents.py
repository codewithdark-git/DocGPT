import base64
from typing import Optional, Tuple, Union, Any
from pathlib import Path
from PIL import Image
import logging
import torch
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from .Pneumonia import Pneumonia
from .Skin_cancer import MelanomaPipeline
from .Eye import Eye
from app.config import get_settings

settings = get_settings()

class DiseaseAgent:
    def __init__(self, img_path: str, task: str):
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        self.image_path = img_path
        self.task = task
        self.prompt = PromptTemplate(
            template="""As a specialist in {task} diagnosis, analyze the provided {task} image and offer a comprehensive 
            report including the diagnosis, severity, and treatment recommendations.\nDiagnosis: {prediction}\nConfidence Level: 
            {confidence}""",
            input_variables=["task", "prediction", "confidence"]
        )
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL_NAME
        )
        
        # Initialize models only when needed
        self._pneumonia_model: Optional[Pneumonia] = None
        self._melanoma_model: Optional[MelanomaPipeline] = None
        self._brain_model: Optional[Any] = None
        self._heart_model: Optional[Any] = None
        self._eye_model: Optional[Any] = None
        
        self.task_to_model = {
            "pneumonia": self.predict_pneumonia,
            "skin_cancer": self.predict_melanoma,
            "brain_tumor": self.predict_brain,
            "heart_disease": self.predict_heart,
            "eye_disease": self.predict_eye,
        }

    def process_image(self) -> Optional[str]:
        """Process the image and encode it to a base64 string."""
        try:
            with Image.open(self.image_path) as image:
                # Convert image to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_bytes = image.tobytes()
                base64_str = base64.b64encode(image_bytes).decode("utf-8")
                return base64_str
        except (IOError, OSError) as e:
            raise ValueError(f"Invalid image file: {e}")
        except Exception as e:
            return f"Error Occur while creating"

    def agent(self) -> Tuple[str, float]:
        """Select the appropriate model for the task and make predictions."""
        if self.task not in self.task_to_model:
            raise ValueError(f"Invalid task: {self.task}. Available tasks are {list(self.task_to_model.keys())}")
        
        return self.task_to_model[self.task]()

    @property
    def pneumonia_model(self) -> Pneumonia:
        if self._pneumonia_model is None:
            self._pneumonia_model = Pneumonia()
        return self._pneumonia_model

    @property
    def melanoma_model(self) -> MelanomaPipeline:
        if self._melanoma_model is None:
            self._melanoma_model = MelanomaPipeline()
        return self._melanoma_model

    @property
    def eye_model(self) -> Eye:
        if self._eye_model is None:
            self._eye_model = Eye()
        return self._eye_model
    
    def predict_eye(self) -> Tuple[str, float]:
        """Predict eye condition."""
        try:
            return self.eye_model.predict_with_torch(self.image_path)
        except Exception as e:
            raise RuntimeError(f"Error predicting Eye: {str(e)}")
        
    def predict_pneumonia(self) -> Tuple[str, float]:
        """Predict pneumonia using the Pneumonia model."""
        try:
            return self.pneumonia_model.predict_with_torch(self.image_path)
        except Exception as e:
            raise RuntimeError(f"Error predicting Pneumonia: {str(e)}")

    def predict_melanoma(self) -> Tuple[str, float]:
        """Predict melanoma using the MelanomaPipeline model."""
        try:
            return self.melanoma_model.predict_with_torch(self.image_path)
        except Exception as e:
            raise RuntimeError(f"Error predicting Melanoma: {str(e)}")

    def predict_brain(self) -> Tuple[str, float]:
        """Predict brain condition."""
        raise NotImplementedError("Brain disease prediction is not yet implemented")

    def predict_heart(self) -> Tuple[str, float]:
        """Predict heart condition."""
        raise NotImplementedError("Heart disease prediction is not yet implemented")

    def response(self) -> dict:
        """Generate a response by combining the image and prediction result."""
        try:
            image_base64 = self.process_image()
            if not image_base64:
                raise ValueError("Failed to process the image.")

            prediction, confidence = self.agent()
            prompt_text = self.prompt.format(
                task=self.task,
                prediction=prediction,
                confidence=confidence
            )
            analysis = self.llm.invoke(prompt_text)
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "analysis": analysis
            }
        except Exception as e:
            return {"error": str(e)}
