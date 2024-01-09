import numpy as np
import polars as pl
from PIL import Image
from glob import glob
import tensorflow as tf
from typer import Argument
from typing import Annotated


def predictimg(
    jenis: Annotated[str, Argument(help="Jenis citra bin atau sdf.")] = "bin",
):
    foil = []
    images = []
    sudut = []

    for gambar in glob(f"aerofoil_images/*_{jenis}_*"):
        fname = gambar.split("/")[-1]
        fname = fname.replace(".jpg", "")
        alpha = fname.split("_")[-1]

        foil.append(fname)
        sudut.append(alpha)
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
        pl.Series(name="sudut", values=sudut).alias("sudut"),
    )

    df.write_csv(f"hasil_prediksi_{jenis}.csv")
