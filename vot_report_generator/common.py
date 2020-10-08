import statistics as st
import matplotlib.pyplot as plt
import pandas as pd

def generate_images(images, link):
    link.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        image.savefig(link / name)

def generate_statistics(sequences):
    table = {}
    for seqname, seq in sequences.items():
        if len(seq) < 2:
            continue
        stats = {}
        stats[f'mean'] = st.mean(seq)
        stats[f'median'] = st.median(seq)
        stats[f'stdev'] = st.stdev(seq)
        table[seqname] = stats
    return table

def timepoints_to_durations(seq):
    res = []
    for idx in range(len(seq) // 2):
        res.append(seq[idx * 2 + 1] - seq[idx * 2])
    return res

def durations_to_fps(seq):
    return [1.0 / elem for elem in seq]
