from fastapi import APIRouter, HTTPException
from app.models import SummarizeRequest, SummarizeResponse
from services.router import predict_with_fallback

router = APIRouter()

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    try:
        summary, model_used, latency = predict_with_fallback(request)

        return SummarizeResponse(
            summary=summary,
            model_used=model_used,
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service temporarily unavailable: {str(e)}")
