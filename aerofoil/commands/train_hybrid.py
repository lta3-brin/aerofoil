import numpy as np
import polars as pl
from PIL import Image
from glob import glob
import tensorflow as tf
from typing import Annotated
from typer import Option, Argument


def trainhybrid(
    jenis: Annotated[str, Argument(help="Jenis citra bin atau sdf.")] = "bin",
    batchsize: Annotated[int, Option(help="Jumlah grup dalam kumpulan dataset.")] = 128,
):
    model = tf.keras.models.load_model(f"aerofoil_checkpoints/model{jenis}")
    model.pop()

    reader = pl.read_csv_batched(f"aerofoil_datasets/{jenis}.csv", batch_size=batchsize)
    batches = reader.next_batches(batchsize)

    for df in batches:
        df = df.select(pl.col("alpha", "cl", "cd", "cm", "img"))

        fitur = df.select(pl.col("img")).to_numpy().flatten()

        images = []
        for gmbr in fitur:
            with Image.open(f"aerofoil_datasets/images/{gmbr}").convert(
                mode="RGB"
            ) as img:
                im = np.asarray(img)
                im = im / 255.0
                images.append(im)

        images = np.asarray(images)
        pred = model.predict(images)

        # label = df.select(pl.col("cl", "cd", "cm")).to_numpy()
