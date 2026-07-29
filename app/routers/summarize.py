from fastapi import APIRouter, HTTPException
from app.models import SummarizeRequest, SummarizeResponse
from services.openai_service import predict_openai

router = APIRouter()

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    try:
        summary, latency = predict_openai(request)

        return SummarizeResponse(
            summary=summary,
            model_used="gpt-4o-mini",
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
