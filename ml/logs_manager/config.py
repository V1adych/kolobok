import os
from dataclasses import dataclass

ACCESS_KEY = os.getenv("YC_ACCESS_KEY")
SECRET_KEY = os.getenv("YC_SECRET_KEY")
BUCKET_NAME = os.getenv("YC_BUCKET_NAME")


@dataclass
class LogsConfig:
    access_key: str = ACCESS_KEY
    secret_key: str = SECRET_KEY
    bucket_name: str = BUCKET_NAME
    endpoint_url: str = "https://storage.yandexcloud.net"
    prefix: str = "logs"
