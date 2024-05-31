import os
import sys
import re
import base64
import platform
from typing import Union, Optional
import csv
import glob
import random
from urllib.parse import urlencode
from urllib.request import urlopen, Request

from pathlib import Path
import importlib


import torch
import torch.nn.functional as F
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

import requests
from PIL import Image
import rimo_storage.cache
from transformers import AutoProcessor, AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection,CLIPModel, CLIPProcessor

sys.path.append(os.path.join(os.path.dirname(__file__), './stable-diffusion-anime-tag-benchmark'))
import ml_danbooru
import WaifuDiffusion_Tagger

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
module_name = "Long-CLIP.model"
model_module = importlib.import_module(module_name)
longclip = getattr(model_module, 'longclip')
longclip_dir = f"{parent_dir}/Long-CLIP"


服务器地址 = f'http://127.0.0.1:7860'

def aesthetic_predictor(img_path):
    this_os,startup_file,device = get_platform()
    SAMPLE_IMAGE_PATH = Path(img_path)
    # load model and preprocessor
    model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(torch.float16).to(device)
    # load image to evaluate
    image = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    # preprocess image
    pixel_values = (
        preprocessor(images=image, return_tensors="pt")
        .pixel_values.to(torch.float16)
        .to(device)
    )
    # predict aesthetic score
    with torch.inference_mode():
        score = model(pixel_values).logits.squeeze().float().cpu().numpy()
    # print result
    return "%.2f" % score

def ml_danbooru标签(image_list: list[Union[str, bytes, os.PathLike]]) -> dict[str, dict[str, float]]:
    超d = {}
    for image in image_list:
        tags = ml_danbooru.get_tags_from_image(Image.open(image), threshold=0.5, keep_ratio=True)
        超d[image] = tags
    return 超d

def WD_tagger(image_path):
    WD = WaifuDiffusion_Tagger.Predictor()
    sorted_general_strings, rating, character_res, general_res = WD.predict(Image.open(image_path),WaifuDiffusion_Tagger.VIT_MODEL_DSV3_REPO,0.02,False,0.1,False)
    ratings = {k:v for k,v in rating.items() if v>=40}
    tags = dict(**character_res,**general_res,**ratings)
    return tags,character_res

def safe_name(s: str):
    return re.sub(r'[\\/:*?"<>|]', lambda m: str(ord(m.group()[0])), s)

def get_platform():
    if platform.system() == "Darwin":
        this_os = "macOS"
        startup_file = "webui.sh"
        if torch.backends.mps.is_available():
            device = "mps"
    elif platform.system() == "Windows":
        this_os = "Windows"
        startup_file = "webui.bat"
    elif platform.system() == "Linux":
        this_os = "Linux"
        startup_file = "webui.bat"
    else:
        this_os = "unknown"
        startup_file = "webui.bat"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return this_os,startup_file,device

def 上网(p, j=None, method='get'):
    r = getattr(requests, method)(p, json=j)
    r.reason = r.text[:4096]
    r.raise_for_status()
    return r.json()

def load_api(url,method='GET'):
    url = 服务器地址+url
    headers = {"accept" :"application/json"}
    data = ""
    data = urlencode(data).encode("utf-8")
    request = Request(url, headers=headers,method=method, data=data)
    try:
        urlopen(request)
    except Exception as e:
        print("An error occurred:", e,"\nThis error can be ignored.")


@rimo_storage.cache.disk_cache(serialize='pickle')
def 缓存上网(p, j=None, method='get'):
    return 上网(p, j=j, method=method)


def txt2img(p: dict):
    r = 缓存上网(f'{服务器地址}/sdapi/v1/txt2img', p, 'post')
    return [base64.b64decode(b64) for b64 in r['images']]


def check_model(要测的模型: list[tuple[Optional[str]]]):
    所有模型 = [i['model_name'] for i in 上网(f'{服务器地址}/sdapi/v1/sd-models')]
    所有VAE = [i['model_name'] for i in 上网(f'{服务器地址}/sdapi/v1/sd-vae')]
    assert set([i[0] for i in 要测的模型]) <= set(所有模型), f'模型只能从{所有模型}中选择，{set([i[0] for i in 要测的模型])-set(所有模型)}不行！'
    assert set([i[1] for i in 要测的模型]) <= set(所有VAE + [None]), f'VAE只能从{所有VAE}或None中选择'

def cos(a, b):
    return a.dot(b) / (a.norm() * b.norm())

clip = None
clip_processor = None
def 图像相似度(img1, img2):
    global clip, clip_processor
    if clip is None:
        clip = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32')
        clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
    inputs = clip_processor(images=[img1, img2], return_tensors="pt") # 画像を前処理してテンソルに変換
    # CLIPモデルで特徴量を抽出
    with torch.no_grad():  # 推論モードで実行（勾配計算を無効化）
        outputs = clip(**inputs)
    img1_embed = outputs.image_embeds[0] # 画像埋め込みを取得
    img2_embed = outputs.image_embeds[1]
    similarity = cos(img1_embed, img2_embed) # コサイン類似度を計算
    return similarity.item()

def truncate_text(text, max_length):
    tokens = text.split(", ")
    truncated_text = []
    current_length = 0
    for token in tokens:
        token_length = len(token) + 2  # 2 is for the added comma and space
        if current_length + token_length > max_length:
            break
        truncated_text.append(token)
        current_length += token_length
    return ", ".join(truncated_text)

def longclip_iANDt(image,text):
    this_os,startup_file,device = get_platform()
    model, preprocess = longclip.load(f"{longclip_dir}/checkpoints/longclip-L.pt", device=device)
    context_length = model.context_length  # これでコンテキスト長を取得します
    text = truncate_text(text, context_length)  # テキストを適切な長さに切り詰めます
    text_emb = longclip.tokenize(text).to(device)
    image_emb = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_emb)
        text_features = model.encode_text(text_emb)
        image_norm = F.normalize(image_features, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)
        cosine_similarity = image_norm @ text_norm.T
    return cosine_similarity

def clip_iANDt(img, text):
    global clip, clip_processor
    if clip is None:
        clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    inputs = clip_processor(text=[text], images=[img], return_tensors="pt", padding=True, truncation=True) # 画像とテキストを前処理してテンソルに変換
    max_length = 77
    if inputs['input_ids'].shape[1] > max_length:
        inputs['input_ids'] = inputs['input_ids'][:, :max_length]
        inputs['attention_mask'] = inputs['attention_mask'][:, :max_length]
    # CLIPモデルで特徴量を抽出
    with torch.no_grad(): # 推論モードで実行（勾配計算を無効化）
        outputs = clip(**inputs) # 画像とテキストの埋め込みを取得
    img_embed = outputs.image_embeds[0]
    text_embed = outputs.text_embeds[0]
    similarity = cos(img_embed, text_embed) # コサイン類似度を計算
    return similarity.item()

clip_text = None
clip_tokenizer = None
def 图文相似度(img, text: str):
    global clip, clip_processor, clip_text, clip_tokenizer
    if clip is None:
        clip = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32')
        clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
    if clip_text is None:
        clip_text = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    outputs_1 = clip(**clip_processor(images=[img], return_tensors="pt"))
    outputs_2 = clip_text(**clip_tokenizer([text], padding=True, return_tensors="pt"))
    return cos(outputs_1.image_embeds[0], outputs_2.text_embeds[0])


def 参数相同(a: dict, b: dict):     # 其实结果是不一样的，但是以前测试的时候忘了指定这个参数
    a = a.copy()
    b = b.copy()
    a['override_settings'].pop('CLIP_stop_at_last_layers', None)
    b['override_settings'].pop('CLIP_stop_at_last_layers', None)
    return a == b

