# SageMaker PyTorch image
# Update base image
# FROM ubuntu:20.04
# FROM 205493899709.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-neo-pytorch:1.9-inf-py3
FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:2.0.0-gpu-py310
# FROM public.ecr.aws/neuron/pytorch-inference-neuron:1.13.1-neuron-py310-sdk2.18.2-ubuntu20.04

# Set the timezone
ENV TZ=Europe/London
ENV DEBIAN_FRONTEND=noninteractive

 # Install necessary Python packages
RUN pip --no-cache-dir install numpy pandas flask gunicorn Pillow
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    nginx \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/tmp* \
    && apt-get clean

# Set environment variables for SageMaker
ENV PATH="/opt/ml/code:${PATH}"
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM encoder.py

# Copy user code
COPY /sam /opt/ml/code
WORKDIR /opt/ml/code

ENTRYPOINT [ "python", "/opt/ml/code/serve" ]