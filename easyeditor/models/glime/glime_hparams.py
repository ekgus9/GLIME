from dataclasses import dataclass
from typing import List, Literal

from ...util.hparams import HyperParams
import yaml


@dataclass
class GLIMEHyperParams(HyperParams):
    # Method
    layers: List[int]

    # Module templates
    mlp_module_tmp: str

    # Statistics
    alg_name: str
    device: int
    model_name: str

    model_parallel: bool = False
    
    epoch: int = 3
    replay_batch: int = 3
    replay_loss_weight: float = 0.1
    sft_loss_weight: float = 1.0
    dpo_loss_weight: float = 1.0
    basis_k: int = 128
    basis_max_rank: int = 1024

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'GLIME') or print(f'GLIMEHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
