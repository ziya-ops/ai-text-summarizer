import logging
from app.models import SummarizeRequest
from services.openai_service import predict_openai
from services.anthropic_service import predict_anthropic

logger = logging.getLogger(__name__)

def predict_with_fallback(request: SummarizeRequest) -> tuple[str, str, float]:
    errors = []

    try:
        logger.info("Attempting OpenAI (primary)")
        summary, latency = predict_openai(request)
        logger.info(f"OpenAI succeeded in {latency:.2f}ms")
        return summary, "gpt-5.4-mini", latency
    except Exception as e:
        error_msg = f"OpenAI failed: {str(e)}"
        logger.warning(error_msg)
        errors.append(error_msg)

    try:
        logger.info("Attempting Anthropic (fallback)")
        summary, latency = predict_anthropic(request)
        logger.info(f"Anthropic succeeded in {latency:.2f}ms")
        return summary, "claude-haiku-4-5", latency
    except Exception as e:
        error_msg = f"Anthropic failed: {str(e)}"
        logger.warning(error_msg)
        errors.append(error_msg)

    error_summary = " | ".join(errors)
    logger.error(f"All providers failed: {error_summary}")
    raise Exception(f"All LLM providers failed: {error_summary}")
