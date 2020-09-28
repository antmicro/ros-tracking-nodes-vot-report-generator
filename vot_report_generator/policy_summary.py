#!/usr/bin/env python3
from jinja2 import FileSystemLoader, Environment
import argparse
from pathlib import Path
import csv
import operator
import matplotlib.pyplot as plt
import statistics
import itertools
from functools import partialmethod
from dataclasses import dataclass
from scipy.signal import savgol_filter
import math
from os import listdir
from os.path import isfile, join
from collections import Counter, defaultdict
import pandas as pd

from generate_images import generate_images

def generate(name, link, fps, idx, test_results):
    env = Environment(
        loader=FileSystemLoader(searchpath="template"))
    template = env.get_template("policy_summary.html")
    images = {}

    ious = []
    for test in test_results.values():
        for testpass in test:
            ious.append(testpass['iou'])
    ious = pd.concat(ious, axis=1)
    images['ious_combined.png'] = ious.plot().get_figure()
    
    generate_images(images, 'output' / link / 'images')
    summary_path = 'output' / link / 'summary.html'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(template.render(
            idx=idx,
            name=name,
            fps=fps))
