from typing import Generator

from datasets import load_dataset
from torch.utils.data import Dataset

from config import DataSetConfig
from models import Language


def get_dataset(config: DataSetConfig) -> Dataset:
    return load_dataset(
        path=config.name,
        name=f'{config.source_lang}-{config.target_lang}',
        split=config.split
    )


def read_ds_by_lang(ds: Dataset, lang: Language) -> Generator:
    for row in ds:
        yield row['translation'][lang]
