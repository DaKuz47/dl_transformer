from pathlib import Path
import json

from pydantic import dataclasses


@dataclasses.dataclass
class DataSetConfig:
    source_lang: str
    target_lang: str
    split: str
    name: str


@dataclasses.dataclass
class Config:
    tokenizers_path: Path
    dataset_config: DataSetConfig


def load_config(path: Path) -> Config:
    with open(path, 'r') as file:
        config_data = json.load(file)
    
    return Config(**config_data)
