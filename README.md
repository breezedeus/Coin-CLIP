# Coin Search Engine by Coin-CLIP 🪙 
Based on OpenAI's **[CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (ViT-B/32)** model, 
we build the **[Coin-CLIP](https://huggingface.co/breezedeus/coin-clip-vit-base-patch32) Model** `breezedeus/coin-clip-vit-base-patch32`,
which is fine-tuned on more than `340,000` coin images using contrastive learning techniques.
**Coin-CLIP** aims to enhance feature extraction capabilities for coin images, thereby achieving more accurate image-based search functionality. This model combines the powerful capabilities of Visual Transformer (ViT) with CLIP's multimodal learning ability, specifically optimized for coin images.

To further simplify the use of the **Coin-CLIP** model, this project provides tools for quickly building a coin image feature search engine.

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

### Building a Vector Search Engine

`coin-clip build-db` can be used to build a vector search engine. It extracts features from all coin images 🪙 in a specified directory and builds a ChromaDB vector search engine.

```bash
$ coin-clip build-db -h
Usage: coin-clip build-db [OPTIONS]

  Extract vectors from a candidate image set and build a search engine based
  on it.

Options:
  -m, --model-name TEXT       Model Name; either local path or huggingface
                              model name
  -d, --device TEXT           ['cpu', 'cuda']; Either 'cpu' or 'gpu', or
                              specify a specific GPU like 'cuda:0'. Default is
                              'cpu'.
  -i, --input-image-dir TEXT  Image Folder to Extract Embeddings  [required]
  -o, --output-db-dir TEXT    Folder where the built search engine is stored.
  -h, --help                  Show this message and exit.
  ```

### Querying
After building the vector search engine with the above command, you can use the `coin-clip retrieve` command to retrieve the coin images 🪙 most similar to a specified coin image.

```bash
$ coin-clip retrieve -h
Usage: coin-clip retrieve [OPTIONS]

  Retrieve images from the search engine, based on the query image.

Options:
  -m, --model-name TEXT  Model Name; either local path or huggingface model
                         name
  -d, --device TEXT      ['cpu', 'cuda']; Either 'cpu' or 'gpu', or specify a
                         specific GPU like 'cuda:0'. Default is 'cpu'.
  --db-dir TEXT          Folder where the built search engine is stored.
  -i, --image-fp TEXT    Image Path to retrieve  [required]
  -h, --help             Show this message and exit.
```
