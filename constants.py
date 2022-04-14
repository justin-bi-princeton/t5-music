# ---------—---------—---------—---------—---------—---------—---------—
# A file containing constants for use by both the training and
# evaluation files. This is to ensure that the constants between the
# two files are the same and that there will never be discrepancies
# between them.
# ---------—---------—---------—---------—---------—---------—---------—

# The maximum input length for the source and target
MAX_LEN = 128

# The seed to give the dataset split
SEED = 11037

# The split between train and test
TEST_SIZE = 0.1

# The directory where the models will be located
MODEL_DIR = './models/'

# Array that takes in the MIDI number modulo'd by 12 as the index
# and outputs the textual representation of the note.
NUM_TO_NOTE = ['C', 'Db', 'D', 'Eb', 'E', 'F',
               'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
