# ----------------------------------------------------------------------
# This creates the initial dataset (song_list.csv) using the original
# lead sheet dataset.
# ----------------------------------------------------------------------
from tqdm import tqdm
import json
import glob
import os
import csv

def main():
    with open('lsd_data.csv', 'w') as csvfile:
        # Write the header to the csv file
        fields = ['path', 'encoded', 'genres']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fields)
        csvwriter.writeheader()

        # Go through all the data files
        for f_name in tqdm(glob.glob(os.path.join('./datasets/', "**/*symbol_key.json"), recursive=True)):
            dict_item = {}

            # Get the path
            path_name = f_name[f_name.index('/event/') + 6:]
            dict_item['path'] = path_name

            # Get the genres
            dir_name = f_name.replace('event', 'xml', 1)
            dir_name = dir_name[:dir_name.rindex('/')] + '/song_info.json'
            with open(dir_name) as xml_f:
                dict_item['genres'] = ';'.join(json.load(xml_f).get('genres'))

            # Get the music
            dict_item['encoded'] = ';'.join(_encode_midi_file(f_name))

            # Write the info
            csvwriter.writerow(dict_item)

# Turn the MIDI melody/harmony into an encoded version
def _encode_midi_file(f_name):
    with open(f_name) as f:
        json_dict = json.load(f)

        # First get all the melody notes and the harmonies into a file
        events = []
        for melody_note in json_dict['tracks']['melody']:
            if not melody_note:
                continue
            # By default, the dataset has middle C listed as 0, which is
            # 60 below the MIDI representation of middle C
            midi = int(melody_note['pitch'] + 60)
            events.append((midi, melody_note['event_on'], 'NB'))
            events.append((midi, melody_note['event_off'], 'NE'))

        for chord in json_dict['tracks']['chord']:
            if not chord:
                continue
            events.append((chord['symbol'], chord['event_on'], 'CB'))
            events.append((chord['symbol'], chord['event_off'], 'CE'))

        # Sort them based on chronological order
        events.sort(key=lambda i: (i[1], i[2]))

        # BPM is used to get the wall clock Time Shifts (eg, a duration of 1.0 is 500 ms in
        # 120 bpm but 400 ms in 150)
        bpm = float(json_dict['metadata'].get('BPM'))
        if bpm == 0:
            bpm = 120

        cur_time = 0
        encoded = []
        for event in events:
            # If the next event doesn't start at the current time offset...
            if cur_time != event[1]:
                # ...turn the duration from the current time to the offset into a TS
                # (max TS of 1000ms)
                time = _beats_to_time(event[1] - cur_time, bpm)
                while time > 1000:
                    encoded.append('TS<1000>')
                    time -= 1000
                encoded.append('TS<' + str(time) + '>')
                cur_time = event[1]
            encoded.append(event[2] + '<' + str(event[0]) + '>')
        return encoded

# Converts the number of beats to the rounded wall time (to the nearest
# ms) given the provided bpm
def _beats_to_time(beats, bpm) -> int:
    # 60 is to convert from minutes to seconds
    return int(round(beats * 60 / bpm * 100)) * 10


if __name__ == "__main__":
    main()
