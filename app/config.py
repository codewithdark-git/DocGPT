from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache

class Settings(BaseSettings):
    PROJECT_NAME: str = "Healthcare AI API"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]

    # Model Paths
    MODEL_PATH: str = "models"
    EYE_MODEL_PATH: str = f"{MODEL_PATH}/eye_model.pth"
    PNEUMONIA_MODEL_PATH: str = f"{MODEL_PATH}/"
    SKIN_MODEL_PATH: str = f"{MODEL_PATH}/skin_disease_model"
    HEART_MODEL_PATH: str = f"{MODEL_PATH}/heart_disease_model"
    BRAIN_MODEL_PATH: str = f"{MODEL_PATH}/brain_tumor_model.pth"
    
    # LLM Configuration
    GROQ_API_KEY: str = ""
    GROQ_MODEL_NAME: str = "mixtral-8x7b-32768"  # or your preferred Groq model
    
    # OpenAI Configuration (backup/alternative)
    OPENAI_API_KEY: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings() 