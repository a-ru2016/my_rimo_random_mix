import sys
import json
import time
import hashlib
import os
import ast

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
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
from common import 上网, 服务器地址,load_api,get_platform
from 评测多标签 import 评测模型

#todo
#自然言語プロンプトをつける

allSteps = 200 #計算回数
save = 200 #何回に一回保存するか
save_last = 0 #最後の何個を保存するか
模型文件夹 = './stable-diffusion-webui/models/Stable-diffusion' #モデル保存場所
#再開用

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_num",
        type=int,
        help="マージするモデル数")
    parser.add_argument(
        "--seed",
        type=int,
        default = 777,
        help="huggingface user name/repository name ユーザー名/リポジトリ名")
    parser.add_argument(
        "--datasets_repo",
        help="huggingface user name/repository name ユーザー名/リポジトリ名")
    parser.add_argument(
        "--auth_token",
        help="huggingface write token 書き込みトークン")
    parser.add_argument(
        "--ver",
        default="XL",
        help="モデルバージョン,WIP")
    parser.add_argument(
        '--amp',
        default="fp16",
        help='計算精度[fp16,fp32]')
    parser.add_argument(
        "-s","--save_steps",
        default="0",
        help='再開するステップ')
    parser.add_argument(
        "-m",'--merge_log_file',
        default="",
        help='merge_log再開させたいファイル名の数字.txt、logからモデルを保存するときに')
    parser.add_argument(
        "-o",'--optimizer_file',
        default="",
        help='optimizer再開させたいファイル名の数字.txt、学習を再開する時に')
    parser.add_argument(
        '--save_only',
        action='store_true',
        help='一度だけ計算して保存する')
    return parser
parser = setup_parser()
args = parser.parse_args()
if args.amp=="fp16":
    amp = np.float16
else:
    amp = np.float32
model_num = args.model_num
merge_log_file = args.merge_log_file
optimizer_file = args.optimizer_file
save_steps = args.save_steps
save_only = args.save_only
seed = args.seed
datasets_repo = args.datasets_repo
auth_token = args.auth_token
commit_message = "烙印融合.py" #アップロード時に書き込むメッセージ
this_os,startup_file,device = get_platform()
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

def 融合识别(s: str) -> str:
    nm={
        "unet": "model.diffusion_model.",
        "clip":"conditioner.embedders.",
    }
    for k, v in nm.items():
        if s.startswith(v):
            n = s.removeprefix(v)
            return f'{k}_{n}'
    return 'r' #全部のキーをマージすることになってるからどうにかして

os.makedirs(模型文件夹+"/output",exist_ok = True)
os.makedirs("log",exist_ok = True)
test_name_log = f'log/Record{int(time.time())}.txt'
log = []
merge_log_name = f'log/merge_log{int(time.time())}.txt'
merge_log = []
optimizer_log_name = f'log/optimizer{int(time.time())}.txt'
steps = 0
识别结果 = set([融合识别(k) for k in all_k])
all_params = []
for i in range(model_num):
    all_params.extend([f"{k}_{i}" for k in 识别结果])

if merge_log_file:
    with open(f"./log/{merge_log_file}") as f:
        s = f.read()
        s = ast.literal_eval(s)
        s = s[save_steps-1]["merge"]
else:
    s = {}
    for k in all_params:
        if int(k[-1]) == model_num-1:
            s[k] = 1
        else:
            s[k] = (1/model_num)*(int(k[-1])+1)

p = {}
for k in all_params:
    if int(k[-1]) == model_num-1:
        p[k] = (1, 1)
    else:
        p[k] = (0.01, 1)

def 烙(**kw):
    global steps#初期化
    test_name = 名字(kw)
    新模型 = {k: 0 for k in all_k}
    old_kw = {k: 0 for k in kw.keys()}
    for i in range(model_num):#merge
        for k in all_k:
            qk = 融合识别(k)
            if (kw[f'{qk}_{i}']-old_kw[f'{qk}_{i}']) > 0:
                weighted_sum = model[i][k] * (kw[f'{qk}_{i}']-old_kw[f'{qk}_{i}'])
                old_kw[f'{qk}_{i+1}'] = kw[f'{qk}_{i}']
            else:
                weighted_sum = model[i][k] * 0
                old_kw[f'{qk}_{i+1}'] = old_kw[f'{qk}_{i}']
            新模型[k] += weighted_sum.astype(np.float16)
    file_path = f'{模型文件夹}/output/{test_name}.safetensors'#save
    save_file(新模型, file_path)
    del 新模型
    load_api('/sdapi/v1/refresh-checkpoints', method='POST')
    acc = 评测模型(test_name, 'sdxl_vae_fp16fix.safetensors', use_tqdm=False, savedata=False, seed=seed, tags_seed=seed)#結果
    load_api('/sdapi/v1/server-restart',method='POST')#webui再起動
    steps += 1 #log
    print(f"Naw steps is {steps}")
    log.append({
        'test_name': test_name,
        'acc': acc,
        'steps': steps,
    })
    merge_log.append({
        'steps': steps,
        'merge': kw,
    })
    print(test_name, acc)#, m.shape)
    with open(test_name_log, 'w', encoding='utf8') as f:
        json.dump(log, f, indent=2)
    with open(merge_log_name, 'w', encoding='utf8') as f:
        json.dump(merge_log, f, indent=2)
    if allSteps==1:
        pass
    elif steps % save == 0 or steps >= allSteps - save_last:#upload
        if auth_token and datasets_repo:
            upload_file(file_path,auth_token)
    else:
        os.remove(file_path)
    return acc

if not save_only:
    subprocess.run(["open", "-a","terminal",re.sub("models/Stable-diffusion","",模型文件夹)+startup_file])
    optimizer = BayesianOptimization(
        f=烙,
        pbounds={k: p[k] for k in all_params},
        random_state=seed,
        #verbose=2.
    )
    optimizer.probe(params={k: s[k] for k in all_params})
    if optimizer_file:
        load_logs(new_optimizer, logs=[optimizer_file]);
    logger = JSONLogger(path=optimizer_log_name)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(
        init_points=4,
        n_iter=allSteps,
    )
    print("done")
elif save_only:
    subprocess.run(["open", "-a","terminal",re.sub("models/Stable-diffusion","",模型文件夹)+startup_file])
    save_last = 1
    allSteps = 1
    烙(**{k: s[k] for k in all_params})