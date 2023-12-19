import os
import typer
import tensorflow as tf
from typing import Annotated
import tensorflow_datasets as tfds

from aerofoil.helpers.image import normalize_img


def trainimg(
    batchsize: Annotated[
        int, typer.Argument(help="Jumlah grup dalam kumpulan dataset.")
    ] = 100,
    showcount: Annotated[bool, typer.Option(help="Tampilkan jumlah dataset?")] = False,
    imgshowtrain: Annotated[
        bool, typer.Option(help="Tampilkan gambar sampel untuk latihan?")
    ] = False,
    imgshowvalid: Annotated[
        bool, typer.Option(help="Tampilkan gambar sampel untuk validasi?")
    ] = False,
):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    (dstrain, dsvalid), info = tfds.load(
        "aerofoil_datasets",
        split=["train", "valid"],
        batch_size=batchsize,
        with_info=True,
        as_supervised=True,
        shuffle_files=True,
    )

    dstrain = dstrain.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    dstrain = dstrain.cache()
    dstrain = dstrain.shuffle(info.splits["train"].num_examples)
    dstrain = dstrain.prefetch(tf.data.AUTOTUNE)

    dsvalid = dsvalid.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    dsvalid = dsvalid.cache()
    dsvalid = dsvalid.prefetch(tf.data.AUTOTUNE)

    if showcount:
        print(f"Jumlah datasets (Train) = {len(dstrain)}")
        print(f"Jumlah datasets (Test) = {len(dsvalid)}")

    if imgshowtrain:
        tfds.show_examples(dstrain, info, is_batched=True)

    if imgshowvalid:
        tfds.show_examples(dsvalid, info, is_batched=True)
