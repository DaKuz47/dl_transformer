from config import load_config
from dataset_utils import get_dataset
from models import Language
from tokenizer import get_tokenizer


if __name__ == '__main__':
    conf = load_config('config.json')

    ds = get_dataset(conf.dataset_config)
    tokenizer = get_tokenizer(conf, ds, Language.RU)

