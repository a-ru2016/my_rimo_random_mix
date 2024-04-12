import sys
import json
import time
import hashlib
import os
import ast

import numpy as np
from bayes_opt import BayesianOptimization
from safetensors.numpy import load_file, save_file
import glob
import argparse
from huggingface_hub import HfApi
from pathlib import Path
from multiprocessing import Process,Pool
from concurrent.futures import ProcessPoolExecutor
import subprocess
import re
from urllib.parse import urlencode
from urllib.request import urlopen, Request

sys.path.append(os.path.join(os.path.dirname(__file__), './stable-diffusion-anime-tag-benchmark'))
from common import 上网, 服务器地址,load_api
from 评测多标签 import 评测模型

allSteps = 1000 #計算回数
save = 500 #何回に一回保存するか
save_last = 2 #最後の何個を保存するか
seed = 777
模型文件夹 = '/Users/naganuma/rimo_random_mix/stable-diffusion-webui-forge/models/Stable-diffusion' #モデル保存場所
model_num = 3 #モデル個数
#再開用
text_file = "merge_log1712691214.txt" #/log内のmerge_logファイル
save_steps = 96 #再開するステップ
save_only = True

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
model = []
model.append(load_model(model_path[0]))
all_k = set(model[0])
for i in range(model_num-1):
    model.append((load_model(model_path[i+1])))
    all_k = all_k & set(model[i+1])

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

os.makedirs(模型文件夹+"/output",exist_ok = True)
os.makedirs("log",exist_ok = True)
记录文件名 = f'log/Record{int(time.time())}.txt'
记录 = []
merge_log_name = f'log/merge_log{int(time.time())}.txt'
merge_log = []
steps = 0
识别结果 = set([融合识别(k) for k in all_k])
all_params = []
for i in range(len(model_path)):
    all_params.extend([f"{k}_{i}" for k in 识别结果])

if text_file:
    with open(f"./log/{text_file}") as f:
        s = f.read()
        s = ast.literal_eval(s)
        s = s[save_steps-1]["merge"]
else:
    s = {k: 1/model_num for k in all_params}

def 烙(**kw):
    global steps#初期化
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
        tmp = 0
        for i in range(len(model_path)):
            kw[f'{qk}_{i}'] *= ratio
    for i in range(model_num):#merge
        for k in all_k:
            qk = 融合识别(k)
            weighted_sum = model[i][k] * kw[f'{qk}_{i}']
            新模型[k] += weighted_sum.astype(np.float16)
    file_path = f'{模型文件夹}/output/{文件名}.safetensors'#save
    save_file(新模型, file_path)
    del 新模型
    load_api('/sdapi/v1/refresh-checkpoints', method='POST')
    结果 = 评测模型(文件名, 'sdxl_vae_fp16fix.safetensors', 32, n_iter=3, use_tqdm=False, savedata=False, seed=seed, tags_seed=seed, 计算相似度=False)
    load_api('/sdapi/v1/server-restart',method='POST')#webui再起動
    m = np.array([]) #loss
    for dd in 结果:
        m = np.append(m, dd['分数'])
    acc = (m > 0.001).sum() / len(m.flatten())
    steps += 1 #log
    print(f"Naw steps is {steps}")
    记录.append({
        '文件名': 文件名,
        'acc': acc,
        'steps': steps,
    })
    merge_log.append({
        'steps': steps,
        'merge': kw,
    })
    print(文件名, acc, m.shape)
    with open(记录文件名, 'w', encoding='utf8') as f:
        json.dump(记录, f, indent=2)
    with open(merge_log_name, 'w', encoding='utf8') as f:
        json.dump(merge_log, f, indent=2)
    if allSteps==1:
        pass
    elif steps % save == 0 or steps >= allSteps - save_last:#upload
        upload_file(file_path,auth_token)
    else:
        os.remove(file_path)
    return acc

if not save_only:
    subprocess.run(["open", "-a","terminal",re.sub("models/Stable-diffusion","",模型文件夹)+"webui.sh"])
    optimizer = BayesianOptimization(
        f=烙,
        pbounds={k: (0.01, 1) for k in all_params},
        random_state=seed,
        #verbose=2.
    )
    optimizer.probe(params={k: s[k] for k in all_params})
    optimizer.maximize(
        init_points=4,
        n_iter=allSteps,
    )
    print("done")
elif save_only:
    subprocess.run(["open", "-a","terminal",re.sub("models/Stable-diffusion","",模型文件夹)+"webui.sh"])
    save_last = 1
    allSteps = 1
    烙(**{k: s[k] for k in all_params})