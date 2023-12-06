# Coin-CLIP 🪙 : Enhancing Coin Image Retrieval with CLIP

 <div align="center">
 <strong>

[[中文]](./README_cn.md) | [English]

 </strong>
</div>

**[Coin-CLIP](https://huggingface.co/breezedeus/coin-clip-vit-base-patch32)** `breezedeus/coin-clip-vit-base-patch32` is built upon 
OpenAI's **[CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (ViT-B/32)** model and fine-tuned on 
a dataset of more than 340,000 coin images using contrastive learning techniques. This specialized model is designed to significantly improve feature extraction for coin images, leading to more accurate image-based search capabilities. Coin-CLIP combines the power of Visual Transformer (ViT) with CLIP's multimodal learning capabilities, specifically tailored for the numismatic domain.

**Key Features:**
- State-of-the-art coin image retrieval;
- Enhanced feature extraction for numismatic images;
- Seamless integration with CLIP's multimodal learning.


To further simplify the use of the **Coin-CLIP** model, this project provides tools for quickly building a coin image retrieval engine.

# Comparison: Coin-CLIP vs. CLIP

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
## Code Examples

### Extract Feature Vectors from Coin Images

```python
from coin_clip import CoinClip

# Automatically download the model from Huggingface
model = CoinClip(model_name='breezedeus/coin-clip-vit-base-patch32')
images = ['examples/10_back.jpg', 'examples/16_back.jpg']
img_feats, success_ids = model.get_image_features(images)
print(img_feats.shape)  # --> (2, 512)
```

> ⚠️ **Note**:
> 
> The above code automatically downloads the [`breezedeus/coin-clip-vit-base-patch32`](https://huggingface.co/breezedeus/coin-clip-vit-base-patch32) model from Huggingface.
If you cannot download automatically, please manually download the model locally, and then initialize `CoinClip` by specifying the local directory of the model through the `model_name` parameter, like `model_name='path/to/coin-clip-vit-base-patch32'`.

## Command line tools

### Building a Vector Retrieval Engine

`coin-clip build-db` can be used to build a vector search engine. It extracts features from all coin images 🪙 in a specified directory and builds a ChromaDB vector search engine.

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

For instance, 

```bash
$ coin-clip build-db -i examples -o coin_clip_chroma.db
```

### Querying
After building the vector search engine with the above command, you can use the `coin-clip retrieve` command to retrieve the coin images 🪙 most similar to a specified coin image.

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

For instance, 

```bash
$ coin-clip retrieve --db-dir coin_clip_chroma.db -i examples/10_back.jpg
```
