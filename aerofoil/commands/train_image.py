import os
import typer
from typing import Annotated
import tensorflow_datasets as tfds

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
    dstrain, infotrain = tfds.load(
        "aerofoil_datasets",
        split="train",
        batch_size=batchsize,
        with_info=True,
        as_supervised=True,
        shuffle_files=True,
    )

    dstest, infotest = tfds.load(
        "aerofoil_datasets",
        split="test",
        batch_size=batchsize,
        with_info=True,
        as_supervised=True,
        shuffle_files=True,
    )

    if showcount:
        print(f"Jumlah datasets (Train) = {len(dstrain)}")
        print(f"Jumlah datasets (Test) = {len(dstest)}")

    if imgshowtrain:
        tfds.show_examples(dstrain, infotrain, is_batched=True)

    if imgshowtest:
        tfds.show_examples(dstest, infotest, is_batched=True)
