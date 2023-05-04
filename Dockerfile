FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && \
    apt install cmake -y && \
    apt install git -y


RUN git clone https://github.com/makaveli10/diffusion.git
WORKDIR diffusion
COPY civit_models civit_models
COPY textual_inversion textual_inversion
COPY civit_lora_models civit_lora_models
RUN bash setup.sh

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--allow-root", "--port=8888", "--no-browser", "--ip=0.0.0.0"]