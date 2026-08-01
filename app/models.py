from pydantic import BaseModel, Field

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=100, max_length=50000)
    max_length: int = Field(default=150, ge=50, le=500)

class SummarizeResponse(BaseModel):
    summary: str
    model_used: str
    latency_ms: float
