# ---------—---------—---------—---------—---------—---------—---------—
# This is the script used for training the model
# ---------—---------—---------—---------—---------—---------—---------—

import argparse
import torch
import wandb

from constants import *
from datasets import load_dataset
from helper import simplify_note_event, simplify_chord_event

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    T5ForConditionalGeneration,
    T5Config
)

# The parser code. Note that you can optionally put all the arguments in
# a text file and prefix it with @ as a singular cmd line arg
def args_setup():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Intake training arguments.'
    )

    # Directory locations
    parser.add_argument('--tokenizer_dir', required=True,
                        help='The directory where the tokenizer is located.')
    parser.add_argument('--dataset_file', default='datasets/song_list.csv',
                        help='What dataset file to use.')
    parser.add_argument('--model_name', default='no_model_name',
                        help='What the output folder/wandb run name are named.')
    parser.add_argument('--checkpoint_dir', default=None,
                        help='The checkpoint directory if you want to continue training.')

    # Model Details
    parser.add_argument('--num_train_epochs', default=10, type=int,
                        help='How many epochs to train.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='The batch size for both train and eval.')
    parser.add_argument('--steps_count', default=400, type=int,
                        help='How often to log, save, and eval.')
    parser.add_argument('--grad_acc', default=64, type=int,
                        help='The number of gradient accumulation steps.')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='The learning rate.')

    # Encoding details
    parser.add_argument('--remove_ts', action='store_true',
                        help='If True, will remove the timeshifts from the encoding.')
    parser.add_argument('--simplify_notes', action='store_true',
                        help='If True, will turn all notes from MIDI numbers to letter representations.')
    parser.add_argument('--simplify_chords', action='store_true',
                        help='If True, will turn all chords into either major or minor triads.')
    parser.add_argument('--remove_end', action='store_true',
                        help='If True, will only include start tokens and not end tokens')

    return parser.parse_args()

def main():

    # Set up the args
    args = args_setup()

    # Set up WandB
    wandb.init(project='t5-music', name=args.model_name)

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

        tokenized_input = tokenizer(
            src, padding='max_length', max_length=MAX_LEN, truncation=True)
        tokenized_target = tokenizer(
            tgt, padding='max_length', max_length=MAX_LEN, truncation=True)

        tokenized_input['decoder_attention_mask'] = tokenized_target['attention_mask']

        # Make sure to change the pad tokens to -100,
        # otherwise they'll mistakenly be account for in the loss
        temp = torch.tensor(tokenized_target['input_ids'])
        temp[temp == tokenizer.pad_token_id] = -100
        tokenized_input['labels'] = list(temp)

        return tokenized_input

    augmented_dataset = dataset.map(
        tokenize, batched=True, batch_size=256, remove_columns=dataset.column_names)
    dataset = augmented_dataset.train_test_split(
        test_size=TEST_SIZE, seed=SEED)

    train_dataset = dataset['train']
    val_dataset = dataset['test']

    train_dataset.set_format(
        'numpy', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(
        'numpy', columns=['input_ids', 'attention_mask', 'labels'])

    # Check whether the model should start from a checkpoint
    # or from scratch
    if args.checkpoint_dir:
        model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_dir)
    else:
        # These are the config details from Google's T5 v1.1 base
        config = T5Config(
            d_ff=2048,
            d_kv=64,
            d_model=768,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            feed_forward_proj='gated-gelu',
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            model_type='t5',
            num_decoder_layers=12,
            num_heads=12,
            num_layers=12,
            output_past=True,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            tie_word_embeddings=False,
            transformers_version='4.15.0',
            use_cache=True,
            vocab_size=tokenizer.vocab_size,
        )
        model = T5ForConditionalGeneration(config)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR + args.model_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # Number of eval steps to keep in GPU (the higher, the mor vRAM used)
        eval_accumulation_steps=2,
        # If I need co compute only loss and not other metrics, setting this to true will use less RAM
        prediction_loss_only=True,
        learning_rate=args.lr,
        evaluation_strategy='steps',  # Run evaluation every eval_steps
        save_steps=args.steps_count,  # How often to save a checkpoint
        save_total_limit=1,  # Number of maximum checkpoints to save
        remove_unused_columns=True,  # Removes useless columns from the dataset
        run_name=args.model_name,  # Wandb run name
        logging_steps=args.steps_count,  # How often to log loss to wandb
        eval_steps=args.steps_count,  # How often to run evaluation on the val_set
        # Whether to load the best model found at each evaluation.
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # Use loss to evaluate best model.
        # Best model is the one with the lowest loss, not highest.
        greater_is_better=False,
        gradient_accumulation_steps=args.grad_acc
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train(resume_from_checkpoint=args.checkpoint_dir)
    trainer.save_model(MODEL_DIR + args.model_name)


if __name__ == "__main__":
    main()
