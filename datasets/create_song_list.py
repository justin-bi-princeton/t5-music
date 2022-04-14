# ----------------------------------------------------------------------
# Create a dataset of every entry from the lead sheet dataset but
# augmented to each of the 12 pitches
# ----------------------------------------------------------------------

import csv
import re
from tqdm import tqdm

# Take a chord and pitch it up one half step
def pitch_up(chord, pitch_amt=1):
    new_chord = list(chord)
    alter_idxs = []

    for _ in range(pitch_amt):
        prev_chord = ''.join(new_chord)

        alter_idxs = []
        # C
        idxs = [i.start() for i in re.finditer('C(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'D'
            alter_idxs.append((idx + 1, 'b'))

        # Db
        idxs = [i.start() for i in re.finditer('Db', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'D'
            alter_idxs.append((idx + 1, 'del'))

        # D
        idxs = [i.start() for i in re.finditer('D(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'E'
            alter_idxs.append((idx + 1, 'b'))

        # Eb
        idxs = [i.start() for i in re.finditer('Eb', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'E'
            alter_idxs.append((idx + 1, 'del'))

        # E
        idxs = [i.start() for i in re.finditer('E(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'F'

        # F
        idxs = [i.start() for i in re.finditer('F(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'G'
            alter_idxs.append((idx + 1, 'b'))

        # Gb
        idxs = [i.start() for i in re.finditer('Gb', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'G'
            alter_idxs.append((idx + 1, 'del'))

        # G
        idxs = [i.start() for i in re.finditer('G(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'A'
            alter_idxs.append((idx + 1, 'b'))

        # Ab
        idxs = [i.start() for i in re.finditer('Ab', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'A'
            alter_idxs.append((idx + 1, 'del'))

        # A
        idxs = [i.start() for i in re.finditer('A(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'B'
            alter_idxs.append((idx + 1, 'b'))

        # Bb
        idxs = [i.start() for i in re.finditer('Bb', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'B'
            alter_idxs.append((idx + 1, 'del'))

        # B
        idxs = [i.start() for i in re.finditer('B(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'C'

        # -----------------------------------------------------------------------------
        # LOWERCASE
        # -----------------------------------------------------------------------------

        # c
        idxs = [i.start() for i in re.finditer('c(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'd'
            alter_idxs.append((idx + 1, 'b'))

        # db
        idxs = [i.start() for i in re.finditer('db', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'd'
            alter_idxs.append((idx + 1, 'del'))

        # d
        idxs = [i.start() for i in re.finditer(
            '(?<![ad])d(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'e'
            alter_idxs.append((idx + 1, 'b'))

        # eb
        idxs = [i.start() for i in re.finditer('eb', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'e'
            alter_idxs.append((idx + 1, 'del'))

        # e
        idxs = [i.start() for i in re.finditer('e(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'f'

        # f
        idxs = [i.start() for i in re.finditer('f(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'g'
            alter_idxs.append((idx + 1, 'b'))

        # gb
        idxs = [i.start() for i in re.finditer('gb', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'g'
            alter_idxs.append((idx + 1, 'del'))

        # g
        idxs = [i.start() for i in re.finditer('g(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'a'
            alter_idxs.append((idx + 1, 'b'))

        # ab
        idxs = [i.start() for i in re.finditer('ab', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'a'
            alter_idxs.append((idx + 1, 'del'))

        # a
        idxs = [i.start() for i in re.finditer('a(?![#bdj])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'b'
            alter_idxs.append((idx + 1, 'b'))

        # bb
        idxs = [i.start() for i in re.finditer('bb', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'b'
            alter_idxs.append((idx + 1, 'del'))

        # b
        idxs = [i.start() for i in re.finditer(
            '(?![A-G|a|c-g])b(?![#b|0-9])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'c'

        for idx, val in sorted(alter_idxs, reverse=True):
            if val == 'del':
                del new_chord[idx]
            else:
                new_chord.insert(idx, val)

    return ''.join(new_chord)

# Pitch up an event if applicable
def pitch_up_event(event):
    # There are 4 types of events.
    # - Note events start w/ N, and can either be text form (C, Bb) or
    #   number form (MIDI 60, 58)
    # - Chord events start w/ C
    # - Timeshift events that start with TS
    # - Mask events that go <mask_id_x>

    # If it starts w/ TS, we don't want to edit it
    if event.startswith('TS'):
        return event

    # Get the content w/in <...>
    content = event[event.index('<') + 1: event.index('>')]

    # If the content stars with mask, we don't want to edit it
    if content.startswith('mask'):
        return event

    # At this point, the event can definitely be pitched up. If the content
    # is a number, increase it by 1, else call pitch up
    if content.isnumeric():
        new_content = str(int(content) + 1)
    else:
        new_content = pitch_up(content)
    return event[:event.index('<') + 1] + new_content + '>'

def main():
    with open('./lsd_data.csv') as f:
        reader = csv.reader(f)
        with open('./song_list.csv', 'w') as wf:
            writer = csv.DictWriter(wf, fieldnames=['encoded'])
            writer.writeheader()
            for row in tqdm(reader):
                if row[0] == 'encoded' or row[0] == '':
                    continue

                writer.writerow({'encoded': row[0]})
                enc_arr = row[1].split(';')
                new_enc = []

                # Augment every entry to the remaining 11 pitches
                for _ in range(11):
                    for item in enc_arr:
                        new_enc.append(pitch_up_event(item))

                    d = {'encoded': ';'.join(new_enc)}
                    writer.writerow(d)

                    enc_arr = new_enc
                    new_enc = []


if __name__ == "__main__":
    main()
