from enum import StrEnum


class Language(StrEnum):
    RU = 'ru'
    EN = 'en'


class SpecialToken(StrEnum):
    UNKNOWN = '[UNK]'
    PADDING = '[PAD]'
    START = '[SOS]'
    END = '[EOS]'
