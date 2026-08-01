import time
from anthropic import Anthropic
from app.config import settings
from app.models import SummarizeRequest
from utils.retry import retry_with_backoff

client = Anthropic(api_key=settings.anthropic_api_key)

@retry_with_backoff(max_retries=3, base_delay=1, max_delay=10)
def predict_anthropic(request: SummarizeRequest) -> tuple[str, float]:
    start_time = time.time()

    prompt = f"Summarize the following text in {request.max_length} words or less:\n\n{request.text}"

    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=request.max_length * 2,
        temperature=0.3,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    summary = response.content[0].text
    latency = (time.time() - start_time) * 1000

    return summary, latency
