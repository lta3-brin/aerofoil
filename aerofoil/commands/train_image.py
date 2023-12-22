import typer
import tensorflow as tf
from typing import Annotated
import tensorflow_datasets as tfds

from aerofoil.helpers.image import normalize_img
from aerofoil.helpers.conv3 import Aerofoil3BN2FC


def trainimg(
    jenis: Annotated[str, typer.Argument(help="Jenis citra bin atau sdf.")] = "sdf",
    epoch: Annotated[int, typer.Option(help="Total pelatihan yang dilakukan.")] = 1000,
    batchsize: Annotated[
        int, typer.Option(help="Jumlah grup dalam kumpulan dataset.")
    ] = 128,
    show_count: Annotated[bool, typer.Option(help="Tampilkan jumlah dataset?")] = False,
    show_img_train: Annotated[
        bool, typer.Option(help="Tampilkan gambar sampel untuk latihan?")
    ] = False,
    show_img_valid: Annotated[
        bool, typer.Option(help="Tampilkan gambar sampel untuk validasi?")
    ] = False,
    show_model_summary: Annotated[
        bool, typer.Option(help="Tampilkan ringkasan tentang model CNN?")
    ] = False,
    lr: Annotated[
        float, typer.Option(help="Learning rate yang dipilih untuk Adam optimizer.")
    ] = 10e-4,
):
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(2024)
    (dstrain, dsvalid), info = tfds.load(
        "aerofoil_datasets",
        split=[f"train{jenis}", f"valid{jenis}"],
        batch_size=batchsize,
        with_info=True,
        as_supervised=True,
        shuffle_files=True,
    )

    dstrain = dstrain.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    dstrain = dstrain.cache()
    dstrain = dstrain.shuffle(info.splits[f"train{jenis}"].num_examples)
    dstrain = dstrain.prefetch(tf.data.AUTOTUNE)

    dsvalid = dsvalid.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    dsvalid = dsvalid.cache()
    dsvalid = dsvalid.prefetch(tf.data.AUTOTUNE)

    image_shape = info.features["image"].shape
    arch = Aerofoil3BN2FC(input_shape=image_shape)
    model = arch.get_model()

    if show_count:
        print(f"Jumlah datasets (Train) = {len(dstrain)}")
        print(f"Jumlah datasets (Test) = {len(dsvalid)}")
    if show_img_train:
        tfds.show_examples(dstrain, info, is_batched=True)
    if show_img_valid:
        tfds.show_examples(dsvalid, info, is_batched=True)
    if show_model_summary:
        model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=tf.keras.metrics.R2Score(name="r2"),
    )

    log = tf.keras.callbacks.TensorBoard(log_dir="./aerofoil_logs")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        verbose=1,
        mode="min",
        monitor="val_loss",
        save_best_only=True,
        filepath="./aerofoil_checkpoints/",
    )

    earlystop = tf.keras.callbacks.EarlyStopping(
        verbose=1, patience=15, monitor="val_loss", restore_best_weights=True
    )

    callbacks = [log, checkpoint, earlystop]
    model.fit(
        dstrain,
        epochs=epoch,
        callbacks=callbacks,
        validation_data=dsvalid,
    )
