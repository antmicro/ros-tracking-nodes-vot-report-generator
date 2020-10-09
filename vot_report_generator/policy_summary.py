#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader 
from scipy.signal import savgol_filter
import pandas as pd

from common import generate_statistics, generate_images


def generate(name, link, fps, idx, test_results, stopwatch_results):
    env = Environment(
        loader=FileSystemLoader(searchpath="template"))
    template = env.get_template("policy_summary.html")
    images = {}

    ious = []
    for test in test_results.values():
        for testpass in test:
            ious.append(testpass['iou'])
    ious_flat = pd.concat(ious, axis=0)
    ious = pd.concat(ious, axis=1)
    images['ious_combined.png'] = ious.plot().get_figure()

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
