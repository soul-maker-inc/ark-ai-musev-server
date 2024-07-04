FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt update -y && apt install --no-install-recommends -y tzdata build-essential ffmpeg libsm6 libxext6 nvidia-cuda-toolkit && rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY requirements_docker.txt /tmp/requirements.txt
RUN pip3 --no-cache-dir install -i https://mirrors.aliyun.com/pypi/simple -r /tmp/requirements.txt

WORKDIR /app
COPY . /app

RUN pip3 --no-cache-dir install -e ./diffusers ./ip_adapter ./clip ./controlnet_aux ./MMCM

ENV PORT 50051
ENV MINIO_ENDPOINT=127.0.0.1:9000
ENV MINIO_ACCESS_KEY=
ENV MINIO_SECRET_KEY=
ENV MINIO_BUCKET=
ENV DEVICE=
ENV SERVICE_PORT=

VOLUME [ "/app/checkpoints" ]
EXPOSE 50051

CMD [ "python3","/app/musev_server.py" ]
