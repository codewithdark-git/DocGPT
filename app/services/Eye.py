import torch
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
from typing import Optional, Tuple
from ..config import get_settings

settings = get_settings()

class Eye:
    MODEL_PATH = settings.EYE_MODEL_PATH
    def __init__(self):
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _get_transforms(self):
        # Define the transformations
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self) -> torch.nn.Module:
        """Load and initialize the PyTorch model."""
        try:
            # Ensure the model path exists
            model_path = Path(self.MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            model = torch.load(model_path, map_location=torch.device("cpu"))
            # Set the model to evaluation mode
            model.eval()
            logging.info("Model loaded successfully")
            return model
        except Exception as e:
            logging.error(f"Model loading error: {str(e)}")

    def predict_with_torch(self, image_path: str) -> Tuple[str, float]:
        """
        Predict using PyTorch model.
        Returns prediction and confidence score.
        """
        try:
            logging.info(f"Starting prediction for image: {image_path}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
            logging.info("Image loaded and preprocessed successfully")

            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                logging.info(f"Raw model output - predicted index: {predicted}, confidence: {confidence}")

            # Convert tensors to Python native types
            prediction = predicted.item()
            confidence_value = confidence.item()
            logging.info(f"Converted to Python types - prediction: {prediction}, confidence: {confidence_value}")

            # Map prediction index to class name
            class_names = ["Normal", "Cataract", "Glaucoma", "Diabetic Retinopathy"]
            prediction_name = class_names[prediction]
            logging.info(f"Mapped to class name: {prediction_name}")

            return prediction_name, confidence_value
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error during model prediction: {str(e)}")

