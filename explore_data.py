from collections import defaultdict

import pandas as pd

data_filename = "chord_data.csv"

df = pd.read_csv(data_filename, header=None)
total_chords = len(df)

notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
chord_types = ['maj', 'min']

chord_counts = defaultdict(int)

for i, row in df.iterrows():
    note_ind = int(row[0])
    chord_type_ind = int(row[1])

    chord = f'{notes[note_ind]} {chord_types[chord_type_ind]}'
    chord_counts[chord] += 1

chord_counts = {k: v for k, v in sorted(chord_counts.items(), key=lambda item: item[1], reverse=True)}
chord_counts_sum = sum(chord_counts.values())

assert total_chords == chord_counts_sum, f'Total chords {total_chords} does not match sum of chord counts {chord_counts_sum}'

for k, v in chord_counts.items():
    print(f'{k}: {v}')

max_count = max(chord_counts.values())
min_count = min(chord_counts.values())

print(f'Most common chord count: {max_count} - Percentage in total dataset: {max_count / total_chords * 100:.2f}%')
print(f'Ideal average occurance of each chord: {1 / len(chord_counts) * 100:.2f}%')