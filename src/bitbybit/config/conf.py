from pydantic_settings import BaseSettings

class Config(BaseSettings):
    epochs: int = 2
    lr : float = 0.01
    batch_size: int = 64


conf = Config()