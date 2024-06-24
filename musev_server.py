import os
import logging
from musev import logger

from concurrent import futures
import logging
import os
import grpc
from grpc_reflection.v1alpha import reflection
from grpc_services import musev_pb2, musev_pb2_grpc, MuseVService

logger.setLevel("DEBUG")
DEVICE = os.environ.get('DEVICE') or "cuda"
SERVICE_PORT = os.environ.get('PORT') or "50051"
MODEL_PATH = os.environ.get('MODEL_PATH') or "./checkpoints"

file_dir = os.path.dirname(__file__)
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "../..")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

def serve():
    logging.info("creating grpc server")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    logging.info("add musev service")
    s1 = MuseVService(MODEL_PATH, DEVICE)
    musev_pb2_grpc.add_MuseVServicer_to_server(s1, server)

    SERVICE_NAMES = (
        musev_pb2.DESCRIPTOR.services_by_name['MuseV'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port("[::]:" + SERVICE_PORT)
    server.start()
    logging.info("Server started, listening on %s",SERVICE_PORT)
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    serve()