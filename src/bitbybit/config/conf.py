from pydantic_settings import BaseSettings

class Config(BaseSettings):
    epochs: int = 2
    lr : float = 0.1
    batch_size: int = 128


conf = Config()
