import statistics as st


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
