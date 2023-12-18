import os
import typer
import polars as pl
from shutil import rmtree
from typing import Annotated


def cleanimg(
    fname: Annotated[str, typer.Argument(
        help="File csv yang diolah.")] = "out.csv",
    re: Annotated[str, typer.Option(
        help="Kolom Reynolds number dalam csv.")] = "re",
    img: Annotated[str, typer.Option(
        help="Kolom image dalam csv.")] = "img",
):
    df = pl.read_csv(fname)
    df = df.filter(pl.col(re) == 103000)

    curdir = os.getcwd()
    fnames = df.select(pl.col(img)).to_numpy().flatten().tolist()

    tmpdir = "tmp"
    os.mkdir(tmpdir)
    for f in fnames:
        try:
            os.rename(f"{curdir}/images/{f}", f"{curdir}/{tmpdir}/{f}")
        except FileNotFoundError:
            pass

    rmtree(f"{curdir}/images/")
    os.rename(tmpdir, "images")
