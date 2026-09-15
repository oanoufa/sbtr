"""Define the sbtr custom model tokenizer."""


from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast
from src import config

workspace_path = config.WORKSPACE_PATH

# Fixed vocab — order fixes the token ids, keep PAD at 0 to match
# CustomBackboneConfig.pad_token_id
VOCAB = {
    "[PAD]": 0,
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "N": 5,
    "-": 6,   # alignment gap, distinct from N
    "[UNK]": 7,
}

tokenizer_model = WordLevel(vocab=VOCAB, unk_token="[UNK]")
raw_tokenizer = Tokenizer(tokenizer_model)
# Split every character into its own token -> 1 token == 1 base, always
raw_tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    pad_token="[PAD]",
    unk_token="[UNK]",
    model_max_length=11648,
)

fast_tokenizer.save_pretrained(f"{workspace_path}/data/model/tokenizer/custom_hiv_tokenizer")