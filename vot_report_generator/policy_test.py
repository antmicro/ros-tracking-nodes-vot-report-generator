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

def find_local_extremums(seq, count, binop):
    batch_size = len(seq) // count
    res = []
    for batch in range(count):
        curr_ext = float('nan')
        ext_idx = float('nan')
        for it in range(batch_size):
            idx = batch_size * batch + it
            val = seq[idx]
            if it == 0 \
                    or binop(curr_ext, val) == val:
                curr_ext = val
                ext_idx = idx 
        res.append((ext_idx, curr_ext))
    return res


def add_extremums_above(ax, vals, input_sequence, binop):
    fig = ax.get_figure()
    width, height = map(int, fig.get_size_inches() * fig.dpi)
    position = ax.get_position()

    annotationsHeight = 0.04
    fig.subplots_adjust(bottom=position.y0,
            top=position.y0 + position.height * (1.0
            - annotationsHeight))

    imgHeight = int(height * annotationsHeight)
    imgShape = image.imread(input_sequence[0]).shape
    imgAspectRatio = imgShape[1] / imgShape [0]
    imgWidth = int(imgAspectRatio * imgHeight)
    spaceSize = 0.07
    spacedImageSize = int(imgWidth * (1.0 + spaceSize))
    nImages = int(width * 0.711) // spacedImageSize
    extremums = find_local_extremums(vals, nImages, binop)
    extr_indexes = [extr[0] for extr in extremums]

    for extr in extremums[:-1]:
        ax.plot(extr[0], extr[1], '*r', markersize=10)

    for it in range(nImages - 1):
        imgIdx = extr_indexes[it]
        lineIdx = len(input_sequence) // nImages * it
        img = image.imread(input_sequence[imgIdx])
        img = cv2.resize(img, dsize=(imgWidth, imgHeight))
        imgPoint = ax.transData.transform_point((lineIdx, 0))
        ax.get_figure().figimage(img, imgPoint[0],
                (1.0 - annotationsHeight - 0.06) * height)

    return mlines.Line2D([], [], color='red', marker='*',
            markersize=10, label='Local extremum', linestyle='None')


def add_images_under(ax, input_sequence):
    fig = ax.get_figure()
    width, height = map(int, fig.get_size_inches() * fig.dpi)
    position = ax.get_position()

    annotationsHeight = 0.04
    fig.subplots_adjust(bottom=position.y0
    # the constant below takes some of the graph's space to be more compact
            + position.height * (annotationsHeight - 0.025), 
    top=position.y0 + position.height)

    imgHeight = int(height * annotationsHeight)
    imgShape = image.imread(input_sequence[0]).shape
    imgAspectRatio = imgShape[1] / imgShape [0]
    imgWidth = int(imgAspectRatio * imgHeight)
    spaceSize = 0.09
    spacedImageSize = int(imgWidth * (1.0 + spaceSize))
    nImages = int(width * 0.711) // spacedImageSize
    for it in range(nImages):
        imgIdx = len(input_sequence) // nImages * it
        ax.axvline(x=imgIdx, linewidth=0.25, color='red')
        img = image.imread(input_sequence[imgIdx])
        img = cv2.resize(img, dsize=(imgWidth, imgHeight))
        imgPoint = ax.transData.transform_point((imgIdx, 0))
        ax.get_figure().figimage(img, imgPoint[0] - imgWidth // 2, 0)

def ious_combined_graph(test_results, input_sequence):
    ious = pd.concat([test['iou'] for test in test_results],
            axis=1, join='inner')
    mean = ious.mean(axis=1)
    ax = ious.plot(legend=False, grid=False, title='Accuracy for'
            + ' every pass and their mean. Images above are local'
            + ' minimums noted by red asterisks on the graph.'
            + ' Images below are frames marked by red vertical'
            + ' lines.')
    for line in ax.lines:
        line.set_color('#C0C0C0')
        line.set_linewidth(1)
    smoothed = [max(0, x) for x in savgol_filter(mean, 15, 3)]
    ax.plot(smoothed, color='blue', linewidth=1)
    ax.set_xlabel('Frame number')
    ax.set_ylabel('IoU')
    ax.grid(color='white', lw = 0.75)

    fig = ax.get_figure()

    add_images_under(ax, input_sequence)
    extr_legend = add_extremums_above(ax, smoothed, input_sequence, min)

    ax.legend(handles=[mlines.Line2D([], [], color='blue',
                          markersize=15, label='Mean (smoothed)'),
                          mlines.Line2D([], [], color='#C0C0C0',
                          markersize=15, label='Raw'),
                          extr_legend
                          ])


    return {'ious_combined.png': fig}


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
