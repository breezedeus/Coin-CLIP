# coding: utf-8

import os
import logging
import click

import numpy as np
import chromadb
from chromadb import Settings
from chromadb.utils.data_loaders import ImageLoader

from .consts import IMAGE_SUFFIX
from .utils import set_logger, read_img
from .chroma_embedding import ChromaEmbeddingFunction

_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}
logger = logging.getLogger(__name__)


def read_img_fps(image_dir):
    # 读取指定目录下的所有图片数据
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_SUFFIX)]
    image_files = [os.path.join(image_dir, f) for f in image_files]
    return image_files


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('build-db')
@click.option(
    '-m',
    '--model-name',
    type=str,
    default='breezedeus/coin-clip-vit-base-patch32"',
    help='模型文件或者名字',
)
@click.option(
    "-d",
    "--device",
    help="['cpu', 'cuda']; 使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为 `cpu`",
    type=str,
    default='cpu',
)
@click.option(
    '-i',
    '--input-image-dir',
    type=str,
    required=True,
    help='Image Folder to Extract Embeddings',
)
@click.option(
    '-o',
    '--output-db-dir',
    type=str,
    default='./coin_clip_chroma.db',
    help='构件好的搜索引擎，存放的文件夹',
)
def extract_embeds_build_db(
    model_name, device, input_image_dir, output_db_dir,
):
    """抽取候选图片集的向量，并基于此构建搜索引擎"""
    set_logger('INFO')

    image_fps = read_img_fps(input_image_dir)
    logger.info(f'Loaded {len(image_fps)} images')

    client = chromadb.PersistentClient(
        path=output_db_dir, settings=Settings(anonymized_telemetry=False)
    )

    embedding_function = ChromaEmbeddingFunction(model_name, device)
    image_loader = ImageLoader()
    collection = client.create_collection(
        name='coin_clip_collection',
        embedding_function=embedding_function,
        # metadata={"hnsw:space": "cosine"},  # l2 is the default
        data_loader=image_loader,
    )

    ids = [str(i) for i in range(len(image_fps))]
    collection.add(ids=ids, uris=image_fps)
    logger.info('%d Items in the collection', collection.count())


@cli.command('retrieve')
@click.option(
    '-m',
    '--model-name',
    type=str,
    default='breezedeus/coin-clip-vit-base-patch32"',
    help='模型文件或者名字',
)
@click.option(
    "-d",
    "--device",
    help="['cpu', 'cuda']; 使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为 `cpu`",
    type=str,
    default='cpu',
)
@click.option(
    '--db-dir', type=str, default='./coin_clip_chroma.db', help='构件好的搜索引擎，存放的文件夹',
)
@click.option(
    '-i', '--image-fp', type=str, required=True, help='Image Path to retrieve',
)
def retrieve(
    model_name, device, db_dir, image_fp,
):
    """抽取候选图片集的向量，并基于此构建搜索引擎"""
    set_logger('INFO')

    client = chromadb.PersistentClient(
        path=db_dir, settings=Settings(anonymized_telemetry=False)
    )

    embedding_function = ChromaEmbeddingFunction(model_name, device)
    image_loader = ImageLoader()
    collection = client.get_collection(
        name='coin_clip_collection',
        embedding_function=embedding_function,
        data_loader=image_loader,
    )
    logger.info('%d Items in the collection', collection.count())  # returns the number of items in the collection

    query_image = np.array(read_img(image_fp))
    retrieved = collection.query(
        query_images=[query_image],
        include=['uris', 'distances', 'data', 'embeddings'],
        n_results=3,
    )
    logger.info('Retrieved URIs: %s', retrieved['uris'][0])
    logger.info('Retrieved Distances: %s', retrieved['distances'][0])


if __name__ == "__main__":
    cli()
