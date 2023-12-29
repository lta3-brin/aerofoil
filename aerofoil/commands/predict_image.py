import numpy as np
import polars as pl
from PIL import Image
from glob import glob
import tensorflow as tf
from typer import Option
from typing import Annotated


def predictimg(
    jenis: Annotated[str, Option(help="Jenis citra bin atau sdf.")] = "bin",
):
    foil = []
    images = []

    for gambar in glob(f"aerofoil_images/*_{jenis}_*"):
        fname = gambar.split("/")[-1]
        fname = fname.replace(".jpg", "")

        foil.append(fname)
        with Image.open(gambar) as img:
            im = np.asarray(img)
            im = im / 255.0
            images.append(im)

    images = np.asarray(images)
    model = tf.keras.models.load_model(f"aerofoil_checkpoints/model{jenis}")
    pred = model.predict(images)

    df = pl.from_numpy(pred, schema=["cl", "cd", "cm"], orient="row")
    df = df.with_columns(
        pl.Series(name="name", values=foil).alias("name"),
    )

    df.write_csv(f"hasil_prediksi_{jenis}.csv")
