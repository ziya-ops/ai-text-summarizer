import time
from openai import OpenAI
from app.config import settings
from app.models import SummarizeRequest

client = OpenAI(api_key=settings.openai_api_key)

def predict_openai(request: SummarizeRequest) -> tuple[str, float]:
    start_time = time.time()

    prompt = f"Summarize the following text in {request.max_length} words or less:\n\n{request.text}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=request.max_length * 2
    )

    summary = response.choices[0].message.content
    latency = (time.time() - start_time) * 1000

    return summary, latency
