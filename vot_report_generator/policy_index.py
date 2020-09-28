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

def generate(name, link, fps, idx, test_results):
    env = Environment(
        loader=FileSystemLoader(searchpath="template"))
    template = env.get_template("policy_index.html")
    index_path = 'output' / link / 'index.html'
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, 'w') as f:
        f.write(template.render(
            idx=idx,
            name=name,
            link=link,
            fps=fps,
            test_results=test_results))
