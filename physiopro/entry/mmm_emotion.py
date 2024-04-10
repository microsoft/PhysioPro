from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment
from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass
from ..dataset import DATASETS
from ..model import MODELS
from ..network import NETWORKS

@configclass
class Config(PythonConfig):
    data: RegistryConfig[DATASETS]
    network: RegistryConfig[NETWORKS]
    decoder_network: RegistryConfig[NETWORKS]
    model: RegistryConfig[MODELS]
    runtime: RuntimeConfig = RuntimeConfig()

def run(config):
    setup_experiment(config.runtime)
    trainset_finetune = config.data.build(dataset_name="train")
    validset_finetune = config.data.build(dataset_name="valid")
    pe_coordination = trainset_finetune.coordination
    network = config.network.build(
        attn_mask = trainset_finetune.attn_mask,
        pe_coordination = pe_coordination,
    )

    model = config.model.build(
        network=network,
        out_size=trainset_finetune.out_size,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
    )
    return model.fit(trainset_finetune, validset_finetune)

if __name__ == "__main__":
    _config = Config.fromcli()
    run(_config)
