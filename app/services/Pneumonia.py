import torch
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
from typing import Optional, Tuple
from app.config import get_settings

settings = get_settings()

class Pneumonia:
    CLASSES = ["Normal", "Pneumonia"]
    MODEL_PATH = settings.PNEUMONIA_MODEL_PATH

    def __init__(self):
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self) -> torch.nn.Module:
        """Load and initialize the PyTorch model."""
        try:
            # Ensure the model path exists
            model_path = Path(self.MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            model = torch.load(model_path, map_location=torch.device("cpu"))
            model.eval()
            logging.info("Model loaded successfully")
            return model

        except Exception as e:
            logging.error(f"Model loading error: {str(e)}")

    @staticmethod
    def _get_transforms():
        """Define image transformations."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_with_torch(self, image_path: str) -> Tuple[str, float]:
        """
        Predict using PyTorch model.
        Returns prediction and confidence score.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            return self.CLASSES[predicted.item()], confidence.item()
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise RuntimeError("Error during model prediction.")
