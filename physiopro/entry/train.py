# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment
from utilsd.experiment import print_config
from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass

from ..dataset import DATASETS, TSDataset, DfDataset
from ..model import MODELS
from ..network import NETWORKS


@configclass
class TrainConfig(PythonConfig):
    data: RegistryConfig[DATASETS]
    network: RegistryConfig[NETWORKS]
    model: RegistryConfig[MODELS]
    runtime: RuntimeConfig = RuntimeConfig()


def run_train(config):
    setup_experiment(config.runtime)
    print_config(config)
    trainset = config.data.build(dataset="train")
    testset = config.data.build(dataset="test", preprocessor=trainset.preprocessor)

    network = config.network.build(
        input_size=trainset.num_variables, max_length=trainset.max_seq_len
    )

    if config.data.type() in [TSDataset, DfDataset] and config.data.task == 'regression':
        out_size = 1
    else:
        out_size = trainset.num_classes

    model = config.model.build(
        network=network,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
        out_size=out_size,
    )
    model.fit(trainset, testset, testset)
    model.predict(trainset, "train")
    model.predict(testset, "test")


if __name__ == "__main__":
    _config = TrainConfig.fromcli()
    run_train(_config)
