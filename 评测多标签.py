import random
import base64
import hashlib
import itertools
from pathlib import Path

import orjson
from PIL import Image
from tqdm import tqdm

from common import 上网, ml_danbooru标签, safe_name, 服务器地址, check_model, 图像相似度, 要测的标签, 参数相同


要测的模型 = [
    ("SDXLAnimePileDriver_v10","sdxl_vae_fp16fix.safetensors"),
    ("animagine-xl-3.1","sdxl_vae_fp16fix.safetensors"),
    ("aniease_v10","sdxl_vae_fp16fix.safetensors"),
    ("4thTailHentaiModel_03","sdxl_vae_fp16fix.safetensors"),
]

sampler = 'DPM++ 2M Karras'
steps = 25
width = 768
height = 1024
cfg_scale = 7

存图文件夹 = Path('out_多标签')
存图文件夹.mkdir(exist_ok=True)

check_model(要测的模型)


if Path('savedata/记录_多标签.json').exists():
    with open('savedata/记录_多标签.json', 'r', encoding='utf8') as f:
        记录 = orjson.loads(f.read())
else:
    记录 = []


def 评测模型(model, VAE, m, n_iter, use_tqdm=True, savedata=True, extra_prompt='', seed=1, tags_seed=0, 计算相似度=True):
    rd = random.Random(tags_seed)
    本地记录 = []
    iterator = range(n_iter)
    if use_tqdm:
        iterator = tqdm(iterator, ncols=70, desc=f'{m}-{model[:10]}')
    for index in iterator:
        标签组 = rd.sample(要测的标签, m)
        标签组 = [i.strip().replace(' ', '_') for i in 标签组]
        参数 = {
            'prompt': f'1 girl, {", ".join(标签组)}'+extra_prompt,
            'negative_prompt': 'worst quality, low quality, blurry, greyscale, monochrome',
            'seed': seed,
            'width': width,
            'height': height,
            'steps': steps,
            'sampler_index': sampler,
            'cfg_scale': cfg_scale,
            'override_settings': {
                'sd_model_checkpoint': model,
                'sd_vae': VAE,
                'CLIP_stop_at_last_layers': 1,
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
            'batch_size': 4,
            'n_iter': 2,
        }
        r = 上网(f'{服务器地址}/sdapi/v1/txt2img', 数量参数 | 参数, 'post')
        图s = [base64.b64decode(b64) for b64 in r['images']]
        md5 = hashlib.md5(str(标签组).encode()).hexdigest()
        for i, b in enumerate(图s):
            with open(存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png'), 'wb') as f:
                f.write(b)
        n = len(图s)
        预测标签 = ml_danbooru标签([存图文件夹 / safe_name(f'{md5}-{i}@{model}×{VAE}@{width}×{height}@{steps}×{sampler}.png') for i in range(n)])

        录 = {
            '分数': [[i.get(j, 0) for j in 标签组] for i in 预测标签.values()],
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
