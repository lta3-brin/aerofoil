import os
import typer
from typing import Annotated
import tensorflow_datasets as tfds

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def trainimg(
    fname: Annotated[str, typer.Argument(
        help="File csv yang diolah.")] = "out.csv",
    re: Annotated[str, typer.Option(
        help="Kolom Reynolds number dalam csv.")] = "re",
):
    ds = tfds.load("aerofoil_datasets", split="train", shuffle_files=True)
    print(ds)