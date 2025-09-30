from functools import lru_cache

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    HOST: str = "127.0.0.1"
    PORT: int = 8080
    DEVELOPMENT_MODE: bool = False


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
