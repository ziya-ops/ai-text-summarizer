from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    anthropic_api_key: str
    openai_base_url: str
    anthropic_base_url: str
    vllm_endpoint: str

    class Config:
        env_file = ".env"

settings = Settings()
