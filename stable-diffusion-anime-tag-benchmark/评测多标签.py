import random
import base64
import hashlib
import itertools
from pathlib import Path
import glob

import orjson
from PIL import Image
from tqdm import tqdm

from common import 上网, ml_danbooru标签, safe_name, 服务器地址, check_model, 图像相似度, 参数相同,WD_tagger

image_path = "/Volumes/TOSHIBAEXT/WEBUI/train/学習用/a_nekonpure-likes/名称未設定フォルダ/anime"
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

存图文件夹 = Path('out_多标签')
存图文件夹.mkdir(exist_ok=True)

#check_model(要测的模型)


if Path('savedata/记录_多标签.json').exists():
    with open('savedata/记录_多标签.json', 'r', encoding='utf8') as f:
        记录 = orjson.loads(f.read())
else:
    记录 = []


def 评测模型(model, VAE, m, n_iter, use_tqdm=True, savedata=True, extra_prompt='', seed=1, tags_seed=0, 计算相似度=True):
    #rd = random.Random(tags_seed)
    本地记录 = []
    iterator = range(n_iter)
    if use_tqdm:
        iterator = tqdm(iterator, ncols=70, desc=f'{m}-{model[:10]}')
    for index in iterator:
        #标签组 = rd.sample(要测的标签, m)
        #标签组 = [i.strip().replace(' ', '_') for i in 标签组]
        path_list = glob.glob(image_path+"/*.jpg")
        path_list = path_list[random.randrange(len(path_list))]
        tag,character_res_in = WD_tagger(path_list)
        标签组 = [k for k in tag.keys()]
        参数 = {
            'prompt': f'score_9, score_8_up, score_7_up, source_anime, {", ".join(标签组)}'+extra_prompt,
            'negative_prompt': 'worst quality, low quality, blurry, greyscale, monochrome,source_furry, source_pony, source_cartoon, score_5_up, score_4_up',
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
        skip = False
        for i in 记录:
            if i['标签组'] == 标签组 and 参数相同(i['参数'], 参数):
                skip = True
                break
        if skip:
            本地记录.append(i)
            continue
        数量参数 = {
            'batch_size': 1,
            'n_iter': 1,
        }
        r = 上网(f'{服务器地址}/sdapi/v1/txt2img', 数量参数 | 参数, 'post')
        图s = [base64.b64decode(b64) for b64 in r['images']]
        md5 = hashlib.md5(str(标签组).encode()).hexdigest()
        for i, b in enumerate(图s):
            with open(存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png'), 'wb') as f:
                f.write(b)
        n = len(图s)
        for i in range(n):
            预测标签,character_res_out = WD_tagger(存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png'))

        tmp = [预测标签.get(j, 0) for j in 标签组]
        tmp.extend([character_res_out.get(j,0) for j in character_res_in.keys()])
        录 = {
            '分数': tmp,
            '总数': n,
            '标签组': 标签组,
            '参数': 参数,
        }
        if 计算相似度:
            相似度 = []
            for a, b in itertools.pairwise([Image.open(存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png')) for i in range(n)]):
                相似度.append(图像相似度(a, b))
            录['相似度'] = 相似度

        本地记录.append(录)
        记录.append(录)
    if savedata:
        with open('savedata/记录_多标签.json', 'wb') as f:
            f.write(orjson.dumps(记录))
    return 本地记录


if __name__ == '__main__':
    for (model, VAE), (m, n_iter) in tqdm([*itertools.product(要测的模型, ((2, 110), (4, 100), (8, 90), (16, 80), (32, 70), (64, 60), (128, 50)))]):
        评测模型(model, VAE, m, n_iter)
