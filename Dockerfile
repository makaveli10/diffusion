FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# # Remove any third-party apt sources to avoid issues with expiring keys.
# RUN rm -f /etc/apt/sources.list.d/*.list

# # Install some basic utilities.
# RUN apt-get update && apt-get install -y \
#     curl \
#     ca-certificates \
#     sudo \
#     git \
#     bzip2 \
#     libx11-6 \
#  && rm -rf /var/lib/apt/lists/*

# RUN apt update

# # install  ssh server
# RUN apt-get install -y openssh-server
# RUN /etc/init.d/ssh start

#  # install python
# RUN apt install software-properties-common -y && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt update

# RUN apt install python3-dev -y && \
#     apt install python-is-python3
    

# # install pip
# RUN apt install python3-pip -y

# Create a working directory.
# RUN mkdir /app
# WORKDIR /app
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