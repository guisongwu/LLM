from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder
from tokenizers.pre_tokenizers import Split, ByteLevel

# --------------------
# Text Data
# --------------------
texts = [
    "hello world",
    "attention is all you need",
    "transformers are powerful",
    "hello!",
    "你好世界"
]

PAT = r"\w+|[^\w\s]+"
# PAT = r"\s" # space


# --------------------
# Build Tokenizer
# --------------------
def build_tokenizer(texts, vocab_size=20):
# def build_tokenizer(texts, vocab_size=30):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(Regex(PAT), behavior="isolated")
    # tokenizer.pre_tokenizer = Split(Regex(PAT), behavior="removed")
    # tokenizer.pre_tokenizer = Split(" ", behavior="removed")
    # tokenizer.pre_tokenizer = Split(" ", behavior="merged_with_previous")
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=ByteLevel.alphabet(),
        # initial_alphabet=ByteLevel.__dict__,
        special_tokens=["[UNK]", "<|endoftext|>"]
    )
    tokenizer.decoder = BPEDecoder()
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer

tokenizer = build_tokenizer(texts)

alphabet = ByteLevel.alphabet()
print(' ' in alphabet)

# --------------------
# Test
# --------------------
s = "transformers are fun!"
enc = tokenizer.encode(s)

print("Text:", s)
print("Token ids:", enc.ids)
print("Decoded:", tokenizer.decode(enc.ids))
print("Vocab size:", tokenizer.get_vocab_size())
print(tokenizer.get_vocab())
