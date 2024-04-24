import os
import json
from tqdm import tqdm
import numpy as np
from cv2 import Canny
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler

def cannyize(image_path):
    img = Image.open(image_path)
    img = np.array(img)
    img = Canny(img, 100, 200)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    canny_img = Image.fromarray(img)
    return canny_img

print('loading model pipeline...')
controlnet = ControlNetModel.from_pretrained('thibaud/controlnet-sd21-canny-diffusers', torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2', controlnet=controlnet, torch_dtype=torch.float16
).to('mps')
pipe.load_lora_weights('model/', weight_name='pytorch_lora_weights.safetensors')
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

img_dir = '../../data/finetune_imgs'
metadata_path = f"{img_dir}/metadata.jsonl"

with open(metadata_path, 'r') as f:
    file = f.read()
metadata = [json.loads(jline) for jline in file.splitlines()]
metadata = {m['file_name']: m['text'] for m in metadata}

print('making canny images...')
image_paths = (f"{img_dir}/{image}" for image in os.listdir(img_dir) if '.png' in image)
canny_images = ((cannyize(image_path), image_path.split('/')[-1]) for image_path in tqdm(image_paths))
image_labels = ((image_name, image, metadata[image_name]) for image, image_name in canny_images)

samples_per = 3
print('generating images...')
for image_name, canny_image, prompt in tqdm(image_labels):
    output = pipe(
        [prompt] * samples_per,
        canny_image,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * samples_per,
        generator=[torch.Generator(device='mps').manual_seed(2) for _ in range(samples_per)],
        num_inference_steps=20,
    )
    for i, image in enumerate(output.images):
        image.save(f"../../data/synthesized_imgs/{image_name.split('.')[0]}_{i}.png")

print('done.')