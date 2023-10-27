# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from shutil import rmtree

from physiopro.entry import train

output_dir = Path("./outputs")


def test_ts_train():
    train.run_train(train.TrainConfig.fromfile(Path(__file__).parent / "configs" / "ts_classification.yml"))
    rmtree("outputs")
    train.run_train(train.TrainConfig.fromfile(Path(__file__).parent / "configs" / "ts_regression.yml"))
    train.run_train(train.TrainConfig.fromfile(Path(__file__).parent / "configs" / "ts_regression_resume.yml"))
    rmtree("outputs")


if __name__ == "__main__":
    rmtree(output_dir)
    test_ts_train()
