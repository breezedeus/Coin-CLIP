# Coin-CLIP 🪙 : 利用 CLIP 技术增强硬币图像检索性能

 <div align="center">
 <strong>

[[English]](./README.md) | [中文]

 </strong>
</div>

**开源 [Coin-CLIP](https://huggingface.co/breezedeus/coin-clip-vit-base-patch32) 模型** `breezedeus/coin-clip-vit-base-patch32` 
在 OpenAI 的 **[CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (ViT-B/32)** 模型基础上，利用对比学习技术在超过 `340,000` 张硬币图片数据上微调得到的。
**Coin-CLIP** 旨在提高模型针对硬币图片的特征提取能力，从而实现更准确的以图搜图功能。该模型结合了视觉变换器（ViT）的强大功能和 CLIP 的多模态学习能力，并专门针对硬币图片进行了优化。

为进一步简化 **Coin-CLIP** 模型的使用流程，本项目提供了快速构建硬币图像检索引擎的工具。

# 效果对比：Coin-CLIP vs. CLIP

#### Example 1 (Left: Coin-CLIP; Right: CLIP)

![1. Coin-CLIP vs. CLIP](./docs/images/3-c.gif)

#### Example 2 (Left: Coin-CLIP; Right: CLIP)

![2. Coin-CLIP vs. CLIP](./docs/images/5-c.gif)

#### More Examples

<details>

<summary>more</summary>

Example 3 (Left: Coin-CLIP; Right: CLIP)
![3. Coin-CLIP vs. CLIP](./docs/images/1-c.gif)

Example 4 (Left: Coin-CLIP; Right: CLIP)
![4. Coin-CLIP vs. CLIP](./docs/images/4-c.gif)

Example 5 (Left: Coin-CLIP; Right: CLIP)
![5. Coin-CLIP vs. CLIP](./docs/images/2-c.gif)

Example 6 (Left: Coin-CLIP; Right: CLIP)
![6. Coin-CLIP vs. CLIP](./docs/images/6-c.gif)

</details>

# Install

```
pip install coin_clip
```

# Usage
## Code examples

### 抽取硬币图片的特征向量

```python
from coin_clip import CoinClip

# 自动从 huggingface 下载模型
model = CoinClip(model_name='breezedeus/coin-clip-vit-base-patch32')
images = ['examples/10_back.jpg', 'examples/16_back.jpg']
img_feats, success_ids = model.get_image_features(images)
print(img_feats.shape)  # --> (2, 512)
```

> ⚠️ **Note**:
> 
> 上面的代码会自动从 Huggingface 下载 [`breezedeus/coin-clip-vit-base-patch32`](https://huggingface.co/breezedeus/coin-clip-vit-base-patch32) 模型。
如果无法自动下载，请手动下载模型到本地，然后初始化 `CoinClip` 时通过 `model_name` 参数指定模型的本地目录，如 `model_name='path/to/coin-clip-vit-base-patch32'`。

## Command line tools

### 构建向量检索引擎

`coin-clip build-db` 可以用来构建向量检索引擎。它对指定目录下的所有硬币图片 🪙 进行特征提取并构建ChromaDB向量检索引擎。

```bash
$ coin-clip build-db -h
Usage: coin-clip build-db [OPTIONS]

  Extract vectors from a candidate image set and build a search engine based
  on it.

Options:
  -m, --model-name TEXT       Model Name; either local path or huggingface
                              model name  [default: breezedeus/coin-clip-vit-
                              base-patch32]
  -d, --device TEXT           ['cpu', 'cuda']; Either 'cpu' or 'gpu', or
                              specify a specific GPU like 'cuda:0'. Default is
                              'cpu'.  [default: cpu]
  -i, --input-image-dir TEXT  Folder with Coin Images to be indexed. [required]
  -o, --output-db-dir TEXT    Folder where the built search engine is stored.
                              [default: ./coin_clip_chroma.db]
  -h, --help                  Show this message and exit.
```

例如：

```bash
$ coin-clip build-db -i examples -o coin_clip_chroma.db
```

### 查询

利用上面的命令构建完向量检索引擎后，可以使用 `coin-clip retrieve` 命令检索与指定硬币图片 🪙 最相似的硬币图片。

```bash
$ coin-clip retrieve -h
Usage: coin-clip retrieve [OPTIONS]

  Retrieve images from the search engine, based on the query image.

Options:
  -m, --model-name TEXT  Model Name; either local path or huggingface model
                         name  [default: breezedeus/coin-clip-vit-base-
                         patch32]
  -d, --device TEXT      ['cpu', 'cuda']; Either 'cpu' or 'gpu', or specify a
                         specific GPU like 'cuda:0'. Default is 'cpu'.
                         [default: cpu]
  --db-dir TEXT          Folder where the built search engine is stored.
                         [default: ./coin_clip_chroma.db]
  -i, --image-fp TEXT    Image Path to retrieve  [required]
  -h, --help             Show this message and exit.
```

例如：

```bash
$ coin-clip retrieve --db-dir coin_clip_chroma.db -i examples/10_back.jpg
```
