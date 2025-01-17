# これは何？

stable diffsionモデルをkeyごとに数値を変えて自動でいい感じにマージしてくれるよ

マージしすぎると壊れるよ

sdxlのみ検証したよ

なんかあったらissuesに書いてね

# 使用方法

コマンドは基本my_rimo_random_mix/で行う

venvを作る

venvの中でpip install -r requirements.txt

my_rimo_random_mix/にstable-diffusion-webuiを入れてモデルも入れる(forgeも可,フォルダ名をstable-diffusion-webuiにする)

Embeddingを入れて使う場合は、评测多标签.pyでnegative_promptと検索してその横にEmbeddingの名前を入れる(もちろんモデルも入れる)

my_rimo_random_mix/にLong-CLIPを入れてモデルも入れる

stable-diffusion-webuiのwebui-user.batのCOMMANDLINE_ARGSに--api --api-server-stopを付ける(macはwebui-user.sh)

stable-diffusion-webuiでtaesdを使うように設定する(オプション、早くなるので推奨)

学習に使いたい画像をimgファイルに入れる

venvの中でpython3 烙印融合.py マージするモデル数の数字(多すぎるとエラーが出てしまうので調整しましょう。２以上じゃないと動かないぞ！)

ひたすら待つ

なんかあったらissuesに書いて

# ステップの途中を保存する方法

python3 烙印融合.py マージするモデル数の数字　-s　保存したいステップ数 -f 再開させたいmerge_logファイル.txt -o

わからないことがあったらpython3 烙印融合.py -hを使うと幸せになれるかもしれない

# 変更したもの

fp16に対応

ステップの途中から再開できる

ステップの途中を保存できる

huggingfaceアップロードをできるようにした

バグ修正

モデルのkeyごとに個別に重みを変えるようにした（これまではin,mid,outで変えてた）

logファイルを生成するようにした

出力の命名規則を変えた

保存回数が膨大だったのを必要最小限にした

WaifuDiffusion_Taggerを使うようにした

long-clipを使うようにした

aesthetic_predictor_v2_5を使うようにした

実際の画像からタグを取得するようにした

mac,windows対応(windows未検証)

# 莉沫酱的随机融合模型！

事情是这样的，最近我测试了不少网上的stable diffusion的动漫风格模型。

其中有1些模型的效果还不错，不过它们各自都有些缺点，于是我就想，那我来做1个天衣无缝的模型好了！

原理是这样的，有2个假设: 

- 合并模型的权重不会引入额外的过拟合。

- 在符合语义方面表现得更好的模型在其他方面也能表现得更好。

嘛，直觉上是这样，第1个假设应该是对的，第2个……我不好说。要是问我为什么我就回答「有人托梦给我」。

总之，这样1来，我们只需要1个特定的指标，然后在各个模型加权得到的空间里暴力搜索，找出1份指标最高的权重就可以了！


## 效果

测试用的prompt是这样: 

```txt
{什么职业}, indoors, twintails, cowboy shot
```

完整的参数是: 

```txt
chef,
indoors, twintails, cowboy shot
Negative prompt: (worst quality, low quality:1.2), from behind
Steps: 50, Sampler: DPM++ 2M SDE Karras, CFG scale: 7, Seed: 1, Size: 640x896, Model hash: c0f0b6c83c, Model: rimochan_random_mix_2.1, VAE hash: 500ea30284, VAE: blessed2.vae.safetensors, Clip skip: 2, Eta: 0.68, Script: X/Y/Z plot, X Type: Prompt S/R, X Values: "chef,scientist,witch,priest,maid,princess,ninja", Y Type: Seed, Y Values: "1,1000", Fixed Y Values: "1, 1000", Version: v1.7.0
```

生成的图片是这样:

![样例.webp](样例.webp)


## 模型下载

Github的LFS超过1G居然要收钱！所以我就把模型传到Civitai了，下载的链接在这里:

<https://civitai.com/models/249129>


## 原理

我们前面说要直接搜出1个指标最高的模型嘛，所以做法是这样:

首先我们需要准备`n`个表现比较好的模型，把他们放在数组`a`里，也就是`[a[1], a[2], ... a[n]]`。

接下来我们还需要另1个长度为`n`的`float`数组`k`，代表给每个模型的乘上这个权重，这样它们融合出来的模型就是: `sum([a[i] * k[i] for i in range(n)])`。

然后我们还要准备1个黑盒函数`f`，用来测试融合出来的模型的指标。我的测指标的仓库是这个: <https://github.com/RimoChan/stable-diffusion-anime-tag-benchmark>，测的指标是每张图32标签下的平均准确率。

好，这样1来问题就变成了找1个`k: list[float]`，使`f(sum([a[i] * k[i] for i in range(n)]))`最大，所以只要挂1个贝叶斯优化让它慢慢跑上几天就好了。

训练<sub>(?)</sub>代码可以参考[超融合.py](超融合.py)和[烙印融合.py](烙印融合.py)。

总之贝叶斯跑完之后，最终确实可以得到1个看起来指标不错的模型。相比起第二名<sub>(blue pencil v10)</sub>，在单个prompt含有2~128标签的测试中，都有2%~27%的提升。

具体的数值大家可以回指标的那个仓库看完整的结果。


## 结束

好，就这样，大家88，我要回去和`1girl`亲热了！

还有我突然想起来天衣无缝，那天衣其实是乳胶衣吧！
