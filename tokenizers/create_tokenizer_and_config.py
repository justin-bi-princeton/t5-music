from transformers import T5Config
from t5_tokenizer_model import SentencePieceUnigramTokenizer

tokenizer_name = 'tokenizer.json'
vocab_filename = 'vocab.txt'
output_folder = './complex_encoding/'

def train_tokenizer():

    tokenizer = SentencePieceUnigramTokenizer(
        unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

    # Train tokenizer (we do it from a file, since these should cover
    # almost all possible tokens one could encounter)
    tokenizer.train(
        files=output_folder + vocab_filename
    )

    # Save files to disk
    tokenizer.save(output_folder + tokenizer_name)
    return tokenizer

def create_config(tokenizer):
    config = T5Config.from_pretrained(
        "google/t5-v1_1-base", vocab_size=tokenizer.get_vocab_size())
    config.save_pretrained(output_folder)

def main():
    tokenizer = train_tokenizer()
    create_config(tokenizer)


if __name__ == "__main__":
    main()
