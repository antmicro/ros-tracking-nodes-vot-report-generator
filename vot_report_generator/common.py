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
