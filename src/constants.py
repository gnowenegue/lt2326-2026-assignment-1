UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
SPECIAL_TOKENS = [
    UNK_TOKEN,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN
]
UNK_ID, PAD_ID, BOS_ID, EOS_ID = range(len(SPECIAL_TOKENS))
LANGUAGES = ['en', 'tr', 'zh']
LANGUAGES_LABEL_MAPPING = {
    'en': 'English',
    'tr': 'Turkish',
    'zh': 'Chinese'
}
RANDOM_SEED = 13
LOCAL_RUN = False
