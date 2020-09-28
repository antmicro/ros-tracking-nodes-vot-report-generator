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

def generate(idx, policy_links):
    env = Environment(
        loader=FileSystemLoader(searchpath="template"))
    template = env.get_template("index.html")
    Path('output').mkdir(exist_ok=True)
    with open("output/index.html", 'w') as f:
        f.write(template.render(
            idx=idx,
            policy_links=policy_links))
