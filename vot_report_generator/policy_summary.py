#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader
from matplotlib import pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import statistics

from common import generate_statistics, generate_images


def iou_bar_graph(ious):
    """Take mean of IoU measurements between
    different tests and draw a bar graph."""
    plt.figure()
    ax = ious.plot.bar()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_xlabel('Test name')
    ax.set_ylabel('Mean IoU')
    ax.set_title('Comparison of avarage IoU between tests')
    fig = ax.get_figure()
    return fig


def iou_size_graph(test_results):
    """Take raw tests results and draw IoU vs bbox size graph."""
    plt.figure()
    dfs = sum([[p for p in t] for t in test_results.values()],
              [])
    df = pd.concat(dfs)
    df['area'] = [x['realWidth'] * x['realHeight'] for index, x
                  in df.iterrows()]
    grouped = df.groupby('area').aggregate(statistics.mean)
    smoothed = [max(0, x) for x in savgol_filter(grouped['iou'], 201, 2)]
    grouped = grouped.assign(smoothed=smoothed)
    grouped_iou = grouped[['iou', 'smoothed']]
    ax = grouped_iou.plot()
    lws = [1, 3]
    for i, l in enumerate(ax.lines):
        plt.setp(l, linewidth=lws[i])
    ax.set_ylabel('Mean IoU')
    ax.set_title('IoU as a function of ground thruth bbox size')
    fig = ax.get_figure()
    return fig


def generate(name, link, fps, idx, test_results, stopwatch_results):
    env = Environment(
        loader=FileSystemLoader(searchpath="templates"))
    template = env.get_template("policy_summary.html")
    images = {}

    ious = []
    avg_per_test = pd.DataFrame()
    for name, test in test_results.items():
        current_ious = []
        for testpass in test:
            ious.append(testpass['iou'])
            current_ious.append(testpass['iou'].mean())
        avg_per_test[name] = current_ious
    avg_per_test = avg_per_test.mean(axis=0)
    ious_flat = pd.concat(ious, axis=0)
    ious = pd.concat(ious, axis=1)
    images['ious_bar.png'] = iou_bar_graph(avg_per_test)
    images['ious_size.png'] = iou_size_graph(test_results)

    iou_table = generate_statistics({'iou': ious_flat})

    generate_images(images, 'output' / link / 'images')
    summary_path = 'output' / link / 'summary.html'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(template.render(
            idx=idx,
            name=name,
            fps=fps,
            iou_table=iou_table))
