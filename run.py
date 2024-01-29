from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from dataset import get_dataset
from llm_wrapper import SaveStrategy, get_llm_wrapper


@dataclass
class RunConfig:
    model_name: str = field(
        default=MISSING,
        metadata={"help": "The name of the model to use. Must be a key in MODEL_CONFIGS."},
    )
    dataset_name: str = field(
        default=MISSING,
        metadata={"help": "The name of the dataset to use. Options are 'leetcode' and 'books'."},
    )
    dataset_path: str = field(
        default=MISSING,
        metadata={
            "help": "The path to the dataset to use. Should be a directory for 'books' and a CSV for 'leetcode'."
        },
    )
    model_size: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The size of the model to use, e.g. '7b'. Overrides the default model size defined "
                " in MODEL_CONFIGS. Only used for HuggingFace models."
            )
        },
    )
    save_strategy: SaveStrategy = field(
        default=SaveStrategy.LOCAL,
        metadata={"help": "The strategy to use for saving model outputs."},
    )
    save_path: Optional[str] = field(
        default="./outputs",
        metadata={"help": "The path to save model outputs to. Only used for SaveStrategy.LOCAL."},
    )


cs = ConfigStore.instance()
cs.store(name="base_config", node=RunConfig)


@hydra.main(version_base=None, config_name="base_config")
def main(config: DictConfig):
    OmegaConf.resolve(config)

    print(f"Run config:\n{'-'*20}\n{OmegaConf.to_yaml(config)}{'-'*20}\n")

    llm = get_llm_wrapper(config)

    dataset = get_dataset(config)

    llm.run_inference(dataset)


if __name__ == "__main__":
    main()
