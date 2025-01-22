from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Healthcare AI API"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]

    # Model Paths
    MODEL_PATH: str = "models"
    EYE_MODEL_PATH: str = f"{MODEL_PATH}/eye_model.pth"
    PNEUMONIA_MODEL_PATH: str = f"{MODEL_PATH}/pneumonia_model.pth"
    SKIN_MODEL_PATH: str = f"{MODEL_PATH}/skin_cancer_model.pth"
    HEART_MODEL_PATH: str = f"{MODEL_PATH}/heart_disease_model"
    BRAIN_MODEL_PATH: str = f"{MODEL_PATH}/brain_tumor_model.pth"
    
    # LLM Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME")

    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings() 