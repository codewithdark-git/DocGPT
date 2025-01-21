from fastapi import APIRouter, HTTPException
from app.schemas.health import HealthCheck

router = APIRouter()

@router.get("/", response_model=HealthCheck)
async def health_check():
    """
    Perform a health check on the API.
    Returns basic information about the API status.
    """
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "services": {
            "database": "up",  # You can add actual database health check here
            "models": "up"     # You can add ML models health check here
        }
    } 