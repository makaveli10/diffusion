# diffusion

## Getting Started

- Install requirements
```bash
 pip install requirements
```

- To easily load civit checkpoints, install the git version of diffusers
```bash
 pip install git+https://github.com/huggingface/diffusers
```

## Docker
- Pull the pre-built docker image
```bash
 docker pull ghcr.io/makaveli10/hello-diffusion:latest
```

- Build your custom docker image
```bash
 docker build . -t hello-diffusion
```

## Notebooks
Already have an idea about the kind of image you want to paint? Have a look at the [Overview](https://github.com/makaveli10/diffusion/blob/main/overview.md) which details the models we have pre-loaded.

Fine-tune a diffusion model to generates images that look like you. For more details and results, refer to the [training overview](https://github.com/makaveli10/diffusion/blob/main/training.md)
