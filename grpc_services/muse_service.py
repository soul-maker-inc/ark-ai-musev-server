from threading import Lock
import os
import sys
import grpc
import logging
import torch
from . import musev_pb2_grpc, musev_pb2
from .musev import Text2Video
from .minio_util import download_minio, upload_to_minio

# 创建一个日志器logger
logger = logging.getLogger("grpc_service")
logger.setLevel("DEBUG")

class MuseVService(musev_pb2_grpc.MuseVServicer):

    def __init__(self, model_path, device):
        logger.info(f"running on {device}")
        self.gen_lock = Lock()
        self.device = device
        self.model_path = model_path

    def unload(self) -> None:
        pass

    def load(self) -> None:
        pass

    def _gc(self):
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def Generate(self, request, context):
        with self.gen_lock:
            srcfile = download_minio(request.conditionImages)
            logger.info(f'downloaded image to {srcfile}')
            output_path = Text2Video(
                device=self.device,
                model_path=self.model_path,
                prompt=request.prompt,
                image_path=srcfile,
                output_dir=request.dest,
                seed=None,
                fps=request.fps,
                w=request.width,
                h=request.height,
                video_len=request.length,
                img_edge_ratio=request.scaleRatio,
                motion_speed=1.0,
            )
            dest = upload_to_minio(output_path, request.dest)
            self._gc()
        return musev_pb2.Img2VideoResponse(result = True, dest = dest)