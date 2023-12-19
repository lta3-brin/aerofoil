"""aerofoil_datasets dataset."""
import polars as pl
import tensorflow as tf
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for aerofoil_datasets dataset."""

    VERSION = tfds.core.Version("1.1.1")
    RELEASE_NOTES = {
        "1.0.0": "Koleksi data latih perdana.",
        "1.1.0": "Pemisahan data train dan test.",
        "1.1.1": "Koreksi nama pemisahan dataset.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(78, 78, 3)),
                    "label": tfds.features.Tensor(shape=(3,), dtype=tf.float64),
                }
            ),
            # If there"s a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://data.mendeley.com/datasets/htw7ymc99n",
        )

    def _split_generators(self, _):
        """Returns SplitGenerators."""

        return {
            "train": self._generate_examples("train"),
            "valid": self._generate_examples("valid"),
        }

    def _generate_examples(self, name):
        df = pl.read_csv("./out.csv")

        if name == "train":
            df = df.sample(fraction=0.8, shuffle=True, seed=2024)
        elif name == "valid":
            df = df.sample(fraction=0.2, shuffle=True, seed=2024)
        else:
            df = df.sample(fraction=0.1, shuffle=True, seed=2024)

        fnames = df.select(pl.col("img")).to_numpy().flatten().tolist()
        coef = df.select(pl.col("cl", "cd", "cm")).to_numpy()

        for idx, f in enumerate(fnames):
            yield idx, {
                "image": f"./images/{f}",
                "label": coef[idx, :],
            }
