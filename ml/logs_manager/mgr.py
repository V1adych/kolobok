from typing import Dict, Any
from pathlib import Path
import datetime
import logging
import io
from PIL import Image
import json

import boto3
import numpy as np

from logs_manager.config import LogsConfig


class LogsMgr:
    def __init__(
        self,
        config: LogsConfig,
    ):
        self.config = config
        self.s3 = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
        )
        self.bucket_name = config.bucket_name
        self.prefix = Path(config.prefix)

        self.logger = logging.getLogger("LogsMgr")

    def _upload_image(self, image: np.ndarray, path: str):
        image = Image.fromarray(image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        self.s3.upload_fileobj(
            Fileobj=buffered,
            Bucket=self.bucket_name,
            Key=path,
        )

    def _upload_json(self, data: Dict[str, Any], path: str):
        self.s3.put_object(
            Body=json.dumps(data).encode("utf-8"),
            Bucket=self.bucket_name,
            Key=path,
        )

    def _upload_txt(self, data: str, path: str):
        self.s3.put_object(
            Body=data.encode("utf-8"),
            Bucket=self.bucket_name,
            Key=path,
        )

    def upload_log(
        self, image: np.ndarray, json_data: Dict[str, Any], metadata: str = ""
    ):
        cur_time = datetime.datetime.now().isoformat()
        directory = self.prefix / cur_time
        self.logger.info(f"Uploading log to {directory}")

        image_path = str(directory / "image.png")
        json_path = str(directory / "data.json")
        metadata_path = str(directory / "metadata.txt")

        self._upload_image(image, image_path)
        self._upload_json(json_data, json_path)
        self._upload_txt(metadata, metadata_path)

        log_path = f"{self.config.endpoint_url}/{self.bucket_name}/{self.prefix}/{cur_time}"

        self.logger.info(f"Log uploaded to {log_path}")
