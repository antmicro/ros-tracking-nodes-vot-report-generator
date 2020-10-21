import statistics as st
import pandas as pd


def generate_images(images, link):
    link.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        image.savefig(link / name)


def generate_statistics(sequences):
    """generates stats for tables"""
    table = {}
    for seqname, seq in sequences.items():
        if len(seq) < 2:
            continue
        stats = {}
        stats['mean'] = st.mean(seq)
        stats['median'] = st.median(seq)
        stats['stdev'] = st.stdev(seq)
        table[seqname] = stats
    return table


def timepoints_to_durations(seq):
    """takes raw stopwatch output, returns durations"""
    res = []
    for idx in range(len(seq) // 2):
        res.append(seq[idx * 2 + 1] - seq[idx * 2])
    return res


def durations_to_fps(seq):
    return [1.0 / elem for elem in seq]


def duration_dataframe_per_frame(tester, stopwatch):
    """takes stopwatch and tester output, returns fps data"""
    l = len(tester[0]['time'])
    table = { name: [0.0] * l for name in stopwatch[0].keys()}
    hits = { name: [0] * l for name in stopwatch[0].keys()}
    for it, ps in enumerate(stopwatch):
        time = tester[it]['time']
        for name, seq in ps.items():
            j = 0
            for idx in range(len(seq) // 2):
                while j < l and time[j] < seq[idx * 2 + 1]:
                    j += 1
                if j == len(time):
                    break
                table[name][j] += \
                        seq[idx * 2 + 1] - seq[idx * 2]
                hits[name][j] += 1
    for name in stopwatch[0]:
        for idx in range(l):
            if hits[name][idx]:
                table[name][idx] /= hits[name][idx]
    return pd.DataFrame(table)
