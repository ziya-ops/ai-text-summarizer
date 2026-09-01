import time
import requests
from app.config import settings
from app.models import SummarizeRequest
from utils.retry import retry_with_backoff

@retry_with_backoff(max_retries=3, base_delay=1, max_delay=10)
def predict_vllm(request: SummarizeRequest) -> tuple[str, float]:
    start_time = time.time()

    prompt = f"Summarize the following text in {request.max_length} words or less:\n\n{request.text}"

    response = requests.post(
        f"{settings.vllm_endpoint}/v1/completions",
        json={
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "prompt": prompt,
            "max_tokens": request.max_length * 2,
            "temperature": 0.3
        },
        timeout=30
    )

    response.raise_for_status()
    result = response.json()

    summary = result["choices"][0]["text"].strip()
    latency = (time.time() - start_time) * 1000

    return summary, latency
