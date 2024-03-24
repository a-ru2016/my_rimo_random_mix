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
from multiprocessing import Process,Pool
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.join(os.path.dirname(__file__), './stable-diffusion-anime-tag-benchmark'))
from common import 上网, 服务器地址
from 评测多标签 import 评测模型
#メモ 1:57スタート
allSteps = 1000 #計算回数
save = 200 #何回に一回保存するか
save_last = 2 #最後の何個を保存するか
seed = 777
模型文件夹 = '/Users/naganuma/rimo_random_mix/stable-diffusion-webui-forge/models/Stable-diffusion'
parallel = 4

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets_repo",
        help="user name/repository name ユーザー名/リポジトリ名")
    parser.add_argument(
        "auth_token",
        help="write token 書き込みトークン")
    parser.add_argument(
        "--ver",
        default="XL",
        help="モデルバージョン,WIP")
    parser.add_argument(
        '--amp',
        default="fp16",
        help='計算精度[fp16,fp32]')
    return parser
parser = setup_parser()
args = parser.parse_args()
if args.amp=="fp16":
    amp = np.float16
else:
    amp = np.float32
datasets_repo = args.datasets_repo
auth_token = args.auth_token
commit_message = "烙印融合.py" #メッセージ

def load_model(filename):
    data = load_file(filename)
    for k, v in data.items():
        data[k] = v.astype(amp)
    return data

model_path = glob.glob(f"{模型文件夹}/*.safetensors")
all_k = load_model(model_path[0])
all_k = set(all_k)
for i in range(len(model_path)-1):
    model = (load_model(model_path[i+1]))
    all_k = all_k & set(model)
    del model

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

def 名字(kw: dict):
    s = sorted(kw.items())
    md5 = hashlib.md5(str(''.join(f'{k}{v:.2f}' for k, v in s)).encode()).hexdigest()
    return f'R3_{md5[:8]}'

def merge(k):
    qk = 融合识别(k)
    weighted_sum = model[k] * kw[f'{qk}_{i}']
    新模型[k] += weighted_sum.astype(np.float16)

steps = 0
def 烙(**kw):
    global steps
    文件名 = 名字(kw)
    新模型 = {}
    for k in all_k:
        新模型[k] = 0
    for k in all_k:#正規化
        qk = 融合识别(k)
        sum=0
        for i in range(len(model_path)):
            sum += kw[f'{qk}_{i}']
        ratio = 1/sum
        for i in range(len(model_path)):
            kw[f'{qk}_{i}'] *= ratio
    for i in range(len(model_path)):#merge
        model = (load_model(model_path[i]))
        print(f"load {i+1}model")
        for k in all_k:
            qk = 融合识别(k)
            weighted_sum = model[k] * kw[f'{qk}_{i}']
            新模型[k] += weighted_sum.astype(np.float16)
        del model
        print(f"kill {i+1}model")
    file_path = f'{模型文件夹}/{文件名}.safetensors'
    save_file(新模型, file_path)
    del 新模型
    上网(f'{服务器地址}/sdapi/v1/refresh-checkpoints', method='post')
    结果 = 评测模型(文件名, 'sdxl_vae_fp16fix.safetensors', 32, n_iter=7, use_tqdm=False, savedata=False, seed=seed, tags_seed=seed, 计算相似度=False)
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
    steps += 1
    print(f"naw steps is{steps}")
    if steps % save == 0 or steps >= allSteps - save_last:
        upload_file(file_path,auth_token)
    else:
        os.remove(file_path)
    return acc

识别结果 = set([融合识别(k) for k in all_k])
all_params = []
for i in range(len(model_path)):
    all_params.extend([f"{k}_{i}" for k in 识别结果])

if __name__ == '__main__':
    optimizer = BayesianOptimization(
        f=烙,
        pbounds={k: (-1, 1) for k in all_params},
        random_state=seed,
        #verbose=2.
    )
    optimizer.probe(params={k: 1/len(model_path) for k in all_params})
    optimizer.maximize(
        init_points=4,
        n_iter=allSteps,
    )
    print("done")