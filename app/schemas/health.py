from pydantic import BaseModel
from typing import Dict, Literal

class HealthCheck(BaseModel):
    status: Literal["healthy", "unhealthy"]
    api_version: str
    services: Dict[str, Literal["up", "down"]]
