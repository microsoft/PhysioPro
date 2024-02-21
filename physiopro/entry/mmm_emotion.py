import torch

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

def load_model(path):
    params = torch.load(path)
    # remove module from name
    keys = list(params.keys())
    for name in keys:
        print(name)
        val=params[name]
        if name.startswith('module.'):
            name = name[7:]
            params[name] = val
            del params['module.'+name]

    # remove network from name
    # keys = list(params.keys())
    # for name in keys:
    #     val=params[name]
    #     if name.startswith('network.'):
    #         name = name[8:]
    #         params[name] = val
    #         del params['network.'+name]
    #     else:
    #         del params[name]

    # remove pos_embed and attn_mask
    if 'pos_embed' in params:
        del params['pos_embed']
    if 'attn_mask' in params:
        del params['attn_mask']
    return params


def run(config):
    setup_experiment(config.runtime)
    trainset_finetune = config.data.build(dataset_name="train")
    validset_finetune = config.data.build(dataset_name="valid")
    pe_coordination = trainset_finetune.coordination
    network = config.network.build(
        attn_mask = trainset_finetune.attn_mask,
        pe_coordination = pe_coordination,
    )
    if config.model.model_path is not None:
        network.load_state_dict(load_model(config.model.model_path), strict=False)

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
