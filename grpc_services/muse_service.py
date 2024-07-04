from threading import Lock
import os
import sys
import grpc
import logging
import torch
from . import musev_pb2_grpc, musev_pb2
from .musev import Text2Video
from .minio_util import download_minio, upload_to_minio
import time
import requests
import json

# 创建一个日志器logger
logger = logging.getLogger("grpc_service")
logger.setLevel("DEBUG")

class MuseVService(musev_pb2_grpc.MuseVServicer):

    def __init__(self, model_path, device):
        logger.info(f"running on {device}")
        self.gen_lock = Lock()
        self.device = device
        self.model_path = model_path
        self.jobs = []
        self.jobs_lock = Lock()

    def unload(self) -> None:
        pass

    def load(self) -> None:
        pass

    def _gc(self):
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def processJob(self, request):
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
                video_len=request.length,
                img_edge_ratio=request.scaleRatio,
                motion_speed=request.motionSpeed,
                n_batch=request.batch,
            )
            dest = upload_to_minio(output_path, request.dest)
            self._gc()
        return dest

    def JobRoutine(self):
        while True:
            nextJob = None
            with self.jobs_lock:
                if len(self.jobs) > 0:
                    nextJob = self.jobs.pop()
            if nextJob != None:
                if len(nextJob.progressCallback) > 0:
                    requests.post(nextJob.progressCallback, data=json.dumps({"stdout":"","stderr":"","progress":0.1}), headers={"Content-Type":"application/json"})
                dest = self.processJob(nextJob.input)
                if len(nextJob.finishCallback) > 0:
                    requests.post(nextJob.finishCallback, data=json.dumps({"type":"video","content":dest}), headers={"Content-Type":"application/json"})
            else:
                time.sleep(1)

    def Generate(self, request, context):
        dest = self.processJob(request)
        return musev_pb2.Img2VideoResponse(result = True, dest = dest)
    
    def GenerateAsync(self, request, context):
        logger.info(f'push incoming request')
        with self.jobs_lock:
            self.jobs.insert(0,request)
        return musev_pb2.Img2VideoResponse(result = True, dest = "")