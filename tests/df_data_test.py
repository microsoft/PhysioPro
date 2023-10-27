# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from shutil import rmtree

from physiopro.entry import train

output_dir = Path("./outputs")


def test_train_classification():
    train.run_train(train.TrainConfig.fromfile(Path(__file__).parent / "configs" / "df_classification.yml"))
    rmtree(output_dir)


def test_train_multiclassification():
    train.run_train(train.TrainConfig.fromfile(Path(__file__).parent / "configs" / "df_multiclassification.yml"))
    rmtree(output_dir)


def test_train_regression():
    train.run_train(train.TrainConfig.fromfile(Path(__file__).parent / "configs" / "df_regression.yml"))
    rmtree(output_dir)


if __name__ == "__main__":
    rmtree(output_dir)
    test_train_classification()
    test_train_multiclassification()
    test_train_regression()
