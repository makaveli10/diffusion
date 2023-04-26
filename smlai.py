#from fastai.data.transforms import get_image_files
import json, gc
from IPython.display import HTML
import glob
import math

from diffusers import DiffusionPipeline, StableDiffusionPipeline
from safetensors.torch import load_file
import torch
import os
import random
from pathlib import Path

import pylab as plt

import os,base64

__all__ = ["load_model", "generate",
           "plt", "torch", "load_civit_ckpt", 
           "convert_civit_lora_safetensors_to_diffusers"]


def convert_civit_lora_safetensors_to_diffusers(
    base_model_path,
    checkpoint_path,
    dump_path,
    LORA_PREFIX_UNET='lora_unet',
    LORA_PREFIX_TEXT_ENCODER='lora_te',
    alpha=0.75,
    from_ckpt=True,
    device="cuda:0",
    to_safetensors=False
    ):
    if os.path.exists(dump_path):
        print("Model already exists loading ...")
        pipeline = load_model(dump_path)
        return model
    
    if from_ckpt:
        pipeline = StableDiffusionPipeline.from_ckpt(base_model_path, torch_dtype=torch.float32)
    else:
        # load base model
        pipeline = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float32)

    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)
    pipeline = pipeline.to(device)
    pipeline.save_pretrained(dump_path, safe_serialization=to_safetensors)
    return pipeline


def load_civit_ckpt(ckpt_link_or_path, model_name):
    """
    Loads a Civitai checkpoint into a stable diffusion pipeline.

    Args:
        ckpt_link_or_path: link of the checkpoint on huggingface or path
                           to the downloaded checkpoint.
    
    Returns:
        StableDiffusionPipeline object.
    """
    pipe = StableDiffusionPipeline.from_ckpt(
        ckpt_link_or_path,
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    pipe.modeldir = model_name
    return pipe


def load_model(model_id):
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.modeldir = model_id
    return pipe

def generate(model_id, prompt, negative_prompt=None, seed=31337, steps=50, N=9, w=512, h=512, guidance_scale=9):
    generators = [torch.Generator(device="cuda").manual_seed(seed + i*512) for i in range(N)]
    if isinstance(model_id, str):
        pipe = DiffusionPipeline.from_pretrained(model_id, custom_pipeline="lpw_stable_diffusion.py", torch_dtype=torch.float16).to("cuda")
    else:
        pipe = model_id
        model_id = pipe.modeldir
    
    images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=guidance_scale, generator=generators, num_images_per_prompt=N, width=w, height=h).images
    
    dirname = Path(Path(model_id).name)/base64.urlsafe_b64encode(os.urandom(16)).decode("ascii")
    dirname.mkdir(parents=True, exist_ok=True)
    with open(dirname/'meta.json', 'w') as f:
        json.dump(dict(
            model_id = model_id,
            prompt = prompt,
            negative_prompt = negative_prompt,
            seed = seed,
            steps = steps,
            w = w,
            h = h,
            guidance_scale = guidance_scale,
        ), f)
    img_paths = [dirname/f'{i}.jpg' for i in range(N)]
    for img,path in zip(images, img_paths): img.save(path)
    del pipe
    return HTML(''.join([f'<img style="float:left; width: 32%; margin:5px;" src="{path}" />' for path in img_paths]))

from IPython.display import HTML
display(HTML('''<style>
.inner_cell div.text_cell_render {
    font-size: 125%;
    overflow: none;
    border-left: 10px solid #ff903b;
    background: #ffe5d0;
    margin-left: -15px;
    padding-left: 15px;
}
</style>'''))