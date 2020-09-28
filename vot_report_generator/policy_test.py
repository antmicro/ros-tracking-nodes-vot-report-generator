#!/usr/bin/env python3
from jinja2 import FileSystemLoader, Environment
from pathlib import Path
import statistics
from scipy.signal import savgol_filter
import math
import pandas as pd
from matplotlib import lines as mlines, image, pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import cv2

from generate_images import generate_images

def ious_combined_graph(test_results, input_sequence):
    ious = pd.concat([test['iou'] for test in test_results],
            axis=1, join='inner')
    mean = ious.mean(axis=1)
    ax = ious.plot(legend=False, grid=True, title='Accuracy for '
            + 'every pass and their mean')
    for line in ax.lines:
        line.set_color('#C0C0C0')
        line.set_linewidth(1)
    smoothed = [max(0, x) for x in savgol_filter(mean, 15, 3)]
    ax.plot(smoothed, color='blue', linewidth=1)
    ax.set_xlabel('Frame number')
    ax.set_ylabel('IoU')
    ax.legend(handles=[mlines.Line2D([], [], color='blue',
                          markersize=15, label='Mean (smoothed)'),
                          mlines.Line2D([], [], color='#C0C0C0',
                          markersize=15, label='Raw')])
    ax.grid(color='red', lw = 0.25)

    fig = ax.get_figure()
    width, height = map(int, fig.get_size_inches() * fig.dpi)

    annotationsHeight = 0.15
    fig.subplots_adjust(bottom=annotationsHeight + 0.08, top=1.0)

    imgHeight = int(height * annotationsHeight)
    imgShape = image.imread(input_sequence[0]).shape
    imgAspectRatio = imgShape[1] / imgShape [0]
    imgWidth = int(imgAspectRatio * imgHeight)
    spaceSize = 0.1
    spacedImageSize = int(imgWidth * (1.0 + spaceSize))
    nImages = max(width - 2 * spacedImageSize, 0) // spacedImageSize
    for it in range(nImages):
        img = image.imread(input_sequence[len(input_sequence) // nImages * it])
        img = cv2.resize(img, dsize=(imgWidth, imgHeight))
        ax.get_figure().figimage(img, width // 8 + it * spacedImageSize, 0)

    return {'ious_combined.png': ax.get_figure()}

def generate(name, test, link, test_results, tests_input_path):
    env = Environment(
        loader=FileSystemLoader(searchpath="template"))
    template = env.get_template("policy_test.html")

    input_sequence_path = tests_input_path / test
    imgs = [f for f in input_sequence_path.glob('*') if f.suffix != '.ann']
    seq_paths = {int(p.stem) - 1 : p for p in imgs}

    images = {}
    images.update(ious_combined_graph(test_results, seq_paths))
    generate_images(images, 'output' / link / 'images')

    summary_path = 'output' / link / 'index.html'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(template.render(
            policyname=name,
            testname=test,
            link=link,
            test_results=test_results))
