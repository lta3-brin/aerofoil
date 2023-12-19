import os
import typer
import tensorflow as tf
from typing import Annotated
import tensorflow_datasets as tfds

from aerofoil.helpers.image import normalize_img

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def trainimg(
    batchsize: Annotated[
        int, typer.Argument(help="Jumlah grup dalam kumpulan dataset.")
    ] = 100,
    showcount: Annotated[bool, typer.Option(help="Tampilkan jumlah dataset?")] = False,
    imgshowtrain: Annotated[
        bool, typer.Option(help="Tampilkan gambar sampel train?")
    ] = False,
    imgshowtest: Annotated[
        bool, typer.Option(help="Tampilkan gambar sampel train?")
    ] = False,
):
    (dstrain, dstest), info = tfds.load(
        "aerofoil_datasets",
        split=["train", "test"],
        batch_size=batchsize,
        with_info=True,
        as_supervised=True,
        shuffle_files=True,
    )

    if showcount:
        print(f"Jumlah datasets (Train) = {len(dstrain)}")
        print(f"Jumlah datasets (Test) = {len(dstest)}")

    if imgshowtrain:
        tfds.show_examples(dstrain, info, is_batched=True)

    if imgshowtest:
        tfds.show_examples(dstest, info, is_batched=True)

    dstrain = dstrain.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    dstrain = dstrain.cache()
    dstrain = dstrain.shuffle(info.splits["train"].num_examples)
    dstrain = dstrain.prefetch(tf.data.AUTOTUNE)

    dstest = dstest.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    dstest = dstest.cache()
    dstest = dstest.prefetch(tf.data.AUTOTUNE)
