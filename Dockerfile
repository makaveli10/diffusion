FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt update -y && \
    apt install cmake -y && \
    apt install git -y


RUN git clone https://github.com/makaveli10/diffusion.git
WORKDIR diffusion
COPY civit_models civit_models
COPY textual_inversion textual_inversion
COPY civit_lora_models civit_lora_models
RUN bash setup.sh
RUN chmod +x start.sh

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ./start.sh