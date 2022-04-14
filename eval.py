# ---------—---------—---------—---------—---------—---------—---------—
# This is the training script that should be used with every
# ---------—---------—---------—---------—---------—---------—---------—

import argparse
import torch

from constants import *
from datasets import load_dataset
from helper import simplify_chord_event, simplify_note_event

from transformers import AutoTokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# The parser code. Note that you can optionally put all the arguments in
# a text file and prefix it with @ as a singular cmd line arg
def args_setup():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Intake evaluation arguments.'
    )

    # Directory locations
    parser.add_argument('--tokenizer_dir', required=True,
                        help='The directory where the tokenizer is located.')
    parser.add_argument('--dataset_file', default='datasets/song_list.csv',
                        help='The dataset file to use.')
    parser.add_argument('--model_dir', default='./',
                        help='The directory to load the model from.')

    # Encoding details
    parser.add_argument('--remove_ts', action='store_true',
                        help='If True, will include the timeshifts in the encoding.')
    parser.add_argument('--simplify_notes', action='store_true',
                        help='If True, will turn all notes from MIDI numbers to letter representations.')
    parser.add_argument('--simplify_chords', action='store_true',
                        help='If True, will turn all chords into either major or minor triads.')
    parser.add_argument('--remove_end', action='store_true',
                        help='If True, will only include start tokens and not end tokens.')

    return parser.parse_args()

def main():

    # Set up the args
    args = args_setup()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # Dataset
    dataset = load_dataset(
        'csv', data_files=args.dataset_file, cache_dir="./cache")['train']

    # Modifies the musical data format into the desired encoding
    def encode(item):
        split = item.split(';')

        src = []
        tgt = []
        i_src = 0
        i_tgt = 0

        # Whether the current event is in a chord or not
        # (so we can determine when to mask them with special ids)
        in_ch = False
        for event in split:

            # In this case, we want to ignore the time shift event
            if args.remove_ts and event.startswith('TS'):
                continue

            # This means it must be a chord
            if event.startswith('C'):

                # In this case, we want to ignore ending events
                if args.remove_end and event.startswith('CE'):
                    continue

                # In this case, we want to simplify the chords
                if args.simplify_chords:
                    event = simplify_chord_event(event)

                # If we encountered a chord and we aren't in a string
                # of chords, then the next mask token must be put
                # into src
                if not in_ch:
                    in_ch = True
                    src.append('<mask_id_' + str(i_src) + '>')
                    i_src += 1

                # Chord events should be appended to tgt
                tgt.append(event)
            else:

                # In this case, we want to ignore ending events
                if args.remove_end and event.startswith('NE'):
                    continue

                # In this case, we want to simplify the chords
                # (ie, change from MIDI to letters)
                if args.simplify_notes and event.startswith('N'):
                    event = simplify_note_event(event)

                # Same logic as if not in_ch
                if in_ch:
                    in_ch = False
                    tgt.append('<mask_id_' + str(i_tgt) + '>')
                    i_tgt += 1

                # Notes and time shifts should go in src
                src.append(event)

        return ';'.join(src), ';'.join(tgt)

    # Tokenize the batch
    def tokenize(batch):
        list_of_tuples = list(map(encode, batch['encoded']))
        src, tgt = list(map(list, list(zip(*list_of_tuples))))

        dct = {
            'src': src,
            'tgt': tgt
        }
        return dct

    augmented_dataset = dataset.map(
        tokenize, batched=True, batch_size=256, remove_columns=dataset.column_names)
    dataset = augmented_dataset.train_test_split(
        test_size=TEST_SIZE, seed=SEED)

    val_dataset = dataset['test']

    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

    model.eval()

    correct_pred = 0
    total_pred = 0  # The number of tokens which need to be predicted
    total_loss = 0

    for i in tqdm(range(len(val_dataset))):

        # Set up the necessary variables to simplify the process
        encoding = val_dataset[i]

        tokenized_input = tokenizer(encoding['src'], padding='max_length',
                                    max_length=MAX_LEN, truncation=True, return_tensors='pt')
        tokenized_target = tokenizer(encoding['tgt'], padding='max_length',
                                     max_length=MAX_LEN, truncation=True, return_tensors='pt')

        input_ids = tokenized_input.input_ids
        attention_mask = tokenized_input.attention_mask
        labels = tokenized_target.input_ids
        labels[labels == tokenizer.pad_token_id] = -100

        desired_len = len(encoding['tgt'].split(';')) + 1

        # The generated output
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            min_length=desired_len,
            max_length=desired_len,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Remove the start token
        gen_no_pad = generated[0][1:]

        # Calculate how many tokens match between the generated
        # output and the target output
        split_tgt = encoding['tgt'].split(';')

        for j in range(len(generated[0][1:])):
            if tokenizer.decode(gen_no_pad[j]) == split_tgt[j]:
                correct_pred += 1

        total_pred += desired_len - 1

        # Calculate the loss between the
        output = model(input_ids=input_ids,
                       attention_mask=attention_mask, labels=labels)
        loss = output.loss
        total_loss += loss.item()

    # Accuracy
    print('Accuracy:', correct_pred / total_pred)

    # Perplexity
    ave_loss = total_loss / len(val_dataset)
    ppl = torch.exp(torch.tensor(ave_loss))
    print('Perplexity:', ppl)


if __name__ == "__main__":
    main()
