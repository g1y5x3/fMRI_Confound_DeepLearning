# pytorch:2.1.0-cuda11.8-cudnn8-runtime
# pytorch:2.1.0-cuda12.1-cudnn8-runtime
FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y vim  && \
    apt-get install -y git  && \
    apt-get -y install python3-pip && \
    pip install mlconfound && \
    pip install wandb && \
    pip install pandas && \
    pip install torchinfo && \
    pip install tqdm && \
    pip install numpy &&\
    pip install nibabel &&\
    pip install pyyaml

ADD data /workspace/data
