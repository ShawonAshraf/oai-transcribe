from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    openai_api_key: SecretStr

    class Config:
        env_file = ".env"


settings = Settings()
