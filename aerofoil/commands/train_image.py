import typer
import tensorflow as tf
from typing import Annotated
import tensorflow_datasets as tfds

from aerofoil.helpers.image import normalize_img
from aerofoil.helpers.conv3 import Aerofoil3BN2FC


def trainimg(
    epoch: Annotated[int, typer.Argument(help="Total pelatihan yang dilakukan.")] = 100,
    batchsize: Annotated[
        int, typer.Option(help="Jumlah grup dalam kumpulan dataset.")
    ] = 128,
    showcount: Annotated[bool, typer.Option(help="Tampilkan jumlah dataset?")] = False,
    imgshowtrain: Annotated[
        bool, typer.Option(help="Tampilkan gambar sampel untuk latihan?")
    ] = False,
    imgshowvalid: Annotated[
        bool, typer.Option(help="Tampilkan gambar sampel untuk validasi?")
    ] = False,
    lr: Annotated[
        float, typer.Option(help="Learning rate yang dipilih untuk Adam optimizer.")
    ] = 10e-4,
):
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

    image_shape = info.features["image"].shape
    arch = Aerofoil3BN2FC(input_shape=image_shape)
    model = arch.get_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=tf.keras.metrics.R2Score(name="r2"),
    )

    model.fit(dstrain, epochs=epoch, validation_data=dsvalid)
