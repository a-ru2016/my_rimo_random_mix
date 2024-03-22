import sys
import json
import time
import hashlib
import os

import numpy as np
from bayes_opt import BayesianOptimization
from safetensors.numpy import load_file, save_file
import glob
import argparse
from huggingface_hub import HfApi
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), './stable-diffusion-anime-tag-benchmark'))
from common import 上网, 服务器地址
from 评测多标签 import 评测模型

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ver",
        default="XL",
        help="モデルバージョン")
    parser.add_argument(
        '--amp',
        default="float16",
        help='計算精度[float16,float32]')
    return parser

parser = setup_parser()
args = parser.parse_args()

if args.amp=="float16":
    amp = np.float16
else:
    amp = np.float32

模型文件夹 = '/Volumes/TOSHIBAEXT/WEBUI/stable-diffusion-webui-rimo/models/Stable-diffusion'

tmp = glob.glob("/Volumes/TOSHIBAEXT/WEBUI/stable-diffusion-webui-rimo/models/Stable-diffusion/*.safetensors")
model = []
def load_fp16_file(filename):
    data = load_file(filename)
    for k, v in data.items():
        data[k] = v.astype(amp)
    return data
for i in range(len(tmp)):
    model.append(load_fp16_file(tmp[i]))
    print(model)
    print(f"load {i+1} model done.")

tmp = set(model[0])
for i in range(len(model)-1):
    tmp = tmp & set(model[i+1])
all_k = tmp

记录文件名 = f'Record{int(time.time())}.txt'
记录 = []

def 融合识别(s: str) -> str:
    nm={
        "unet": "model.diffusion_model.",
        "clip":"conditioner.embedders.",
    }
    for k, v in nm.items():
        if s.startswith(v):
            n = s.removeprefix(v)
            return f'{k}_{n}'
    return 'r'

def upload_file(file_path, token):
          file_name = "モデル/"+Path(file_path).name
          print(f"Uploading {file_name} to https://huggingface.co/datasets/{datasets_repo}")
          print(f"Please wait...")

          api.upload_file(
              path_or_fileobj=file_path,
              path_in_repo=file_name,
              repo_id=datasets_repo,
              repo_type="dataset",
              commit_message=commit_message,
              token=token,
          )
          print(f"Upload success, located at https://huggingface.co/datasets/{datasets_repo}/blob/main/{file_name}\n")
api = HfApi()
datasets_repo = "aru2016/model"
commit_message = "烙印融合.py"
auth_token = "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"

def 名字(kw: dict):
    s = sorted(kw.items())
    md5 = hashlib.md5(str(''.join(f'{k}{v:.2f}' for k, v in s)).encode()).hexdigest()
    return f'R3_{md5[:8]}'


def 烙(**kw):
    文件名 = 名字(kw)
    新模型 = {}
    for k in all_k:
        qk = 融合识别(k)
        weighted_sum = 0
        for i, mdl in enumerate(model):
            weighted_sum += mdl[k] * kw[f'{qk}_{i}']
        新模型[k] = weighted_sum.astype(np.float16)
    file_path = f'{模型文件夹}/{文件名}.safetensors'
    save_file(新模型, file_path)
    upload_file(file_path,auth_token)
    del 新模型
    上网(f'{服务器地址}/sdapi/v1/refresh-checkpoints', method='post')
    结果 = 评测模型(文件名, 'sdxl_vae_fp16fix.safetensors', 32, n_iter=80, use_tqdm=False, savedata=true, seed=22987, tags_seed=2223456, 计算相似度=False)
    m = []
    for dd in 结果:
        m.extend(dd['分数'])
    mm = np.array(m)
    acc = (mm > 0.001).sum() / len(mm.flatten())
    记录.append({
        '文件名': 文件名,
        'acc': acc,
    })
    print(文件名, acc, mm.shape)
    with open(记录文件名, 'w', encoding='utf8') as f:
        json.dump(记录, f, indent=2)
    os.remove(file_path)
    return acc

识别结果 = set([融合识别(k) for k in all_k])
all_params = []
for i, mdl in enumerate(model):
    all_params.extend([f"{k}_{i}" for k in 识别结果])

optimizer = BayesianOptimization(
    f=烙,
    pbounds={k: (-1, 1) for k in all_params},
    random_state=777,
    #verbose=2.
)
optimizer.probe(params={k: 0 for k in all_params})
optimizer.maximize(
    init_points=4,
    n_iter=1000,
)
print("done")