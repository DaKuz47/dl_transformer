from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import Dataset

from config import Config
from dataset_utils import read_ds_by_lang
from models import Language, SpecialToken


def get_tokenizer(config: Config, ds: Dataset, lang: Language) -> Tokenizer:
    tokenizer_path = config.tokenizers_path.joinpath(lang)

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token=SpecialToken.UNKNOWN))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=[token for token in SpecialToken],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(read_ds_by_lang(ds, lang), trainer)
        print(tokenizer_path)
        tokenizer.save(str(tokenizer_path), pretty=True)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer
