import random
import base64
import hashlib
import itertools
from pathlib import Path
import glob

import orjson
from PIL import Image
from tqdm import tqdm
import numpy as np
import os

from common import 上网, ml_danbooru标签, safe_name, 服务器地址, check_model, 图像相似度,WD_tagger,aesthetic_predictor,longclip_iANDt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
image_path = f"{parent_dir}/img"
要测的模型 = [
    ("4thTailHentaiModel_03","sdxl_vae_fp16fix.safetensors"),
    ("aniease_v10","sdxl_vae_fp16fix.safetensors"),
    ("animagine-xl-3.1","sdxl_vae_fp16fix.safetensors"),
    ("animeAntifreezingSolutionXL_v10","sdxl_vae_fp16fix.safetensors"),
    ("animeconfettitunexl_con5tixl","sdxl_vae_fp16fix.safetensors"),
    ("animeIllustDiffusion_v08","sdxl_vae_fp16fix.safetensors"),
    ("autismmixSDXL_autismmixConfetti","sdxl_vae_fp16fix.safetensors"),
    ("heartOfAppleXL_v10","sdxl_vae_fp16fix.safetensors"),
    ("SDXLAnimeBulldozer_v20","sdxl_vae_fp16fix.safetensors"),
    ("SDXLAnimePileDriver_v10","sdxl_vae_fp16fix.safetensors"),
    ("swampmachine_v20","sdxl_vae_fp16fix.safetensors"),
    ("tPonynai3_v20","sdxl_vae_fp16fix.safetensors"),
    ("vivaldixlOpenDalle3_v10","sdxl_vae_fp16fix.safetensors"),
]

sampler = 'DPM++ 2M Karras'
steps = 20
width = 768
height = 1024
cfg_scale = 7

output_path = Path('out_多标签')
output_path.mkdir(exist_ok=True)

#check_model(要测的模型)


if Path('savedata/记录_多标签.json').exists():
    with open('savedata/记录_多标签.json', 'r', encoding='utf8') as f:
        记录 = orjson.loads(f.read())
else:
    记录 = []


def 评测模型(model, VAE,  use_tqdm=True, savedata=True, extra_prompt='', seed=1, tags_seed=0):
    本地记录 = []
    path_list = glob.glob(image_path+"/*.jpg")
    path_list.extend(glob.glob(image_path+"/*.png"))
    if use_tqdm:
        iterator = tqdm(iterator, ncols=70, desc=f'{m}-{model[:10]}')
    for test_img in path_list:
        tag_in,character_res_in = WD_tagger(test_img)
        tag_in_key = [k for k in tag_in.keys()]#プロンプトのリスト
        setting = {
            'prompt': f'score_9, score_8_up, score_7_up, source_anime, {", ".join(tag_in_key)}'+extra_prompt,
            'negative_prompt': 'negativeXL_D,unaestheticXL_AYv1,worst quality,unaestheticXL_Sky3.1,unaestheticXLv1,unaestheticXLv31, low quality, blurry, greyscale, monochrome,source_furry, source_pony, source_cartoon, score_5_up, score_4_up',
            'seed': seed,
            'width': width,
            'height': height,
            'steps': steps,
            'sampler_index': sampler,
            'cfg_scale': cfg_scale,
            'override_settings': {
                'sd_model_checkpoint': model,
                #'sd_vae': VAE,
                'CLIP_stop_at_last_layers': 2,
            },
        }
        数量setting = {
            'batch_size': 1,
            'n_iter': 1,
        }
        r = 上网(f'{服务器地址}/sdapi/v1/txt2img', 数量setting | setting, 'post')#t2i
        图s = [base64.b64decode(b64) for b64 in r['images']]
        md5 = hashlib.md5(str(tag_in_key).encode()).hexdigest()
        png_name = safe_name(f'{[j for j in character_res_in.keys()]}_{model}-{md5}')
        extension = ".png"
        for i, b in enumerate(图s):#save
            with open(output_path / f"{png_name}{extension}", 'wb') as f:
                f.write(b)
        
        tag_out,character_res_out = WD_tagger(f"{output_path}/{png_name}{extension}")#score
        WD = 0
        for i in tag_in_key:
            if tag_out.get(i):
                WD += 1/len(tag_in_key)
            else:
                WD += -1/len(tag_in_key)
        for i in character_res_in.keys():
            if character_res_out.get(i):
                WD += 1/len(tag_in_key)
            else:
                WD += -1/len(tag_in_key)
        aesthetic_score = float(aesthetic_predictor(f"{output_path}/{png_name}{extension}"))/10
        longclip_score = float(longclip_iANDt(Image.open(f"{output_path}/{png_name}{extension}"),setting["prompt"]))
        
        录 = {
            'WD': WD,
            'aesthetic': aesthetic_score,
            'longclip': longclip_score,
            'setting': setting,
        }
        本地记录.append(录)
        
    acc = 0
    for i in ["WD","aesthetic","longclip"]:
        tmp = 0
        for dd in 本地记录:
            tmp += (dd[i])
        acc += tmp/len(path_list)
    acc = acc/3
    
    if savedata:
        with open('savedata/记录_多标签.json', 'wb') as f:
            f.write(orjson.dumps(本地记录))
    return acc

