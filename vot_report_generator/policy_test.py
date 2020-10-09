#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader
from scipy.signal import savgol_filter
import pandas as pd
from matplotlib import cm, image, lines as mlines, pyplot as plt
import cv2
import numpy as np

from common import generate_statistics, generate_images, \
        timepoints_to_durations, durations_to_fps


def find_interval_extremums(seq, count, binop):
    """find {count} extremums on interval
    on {seq} where ordering is defined by
    {binop}"""
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
    """take graph and add frames of extremums
    above the graph"""
    fig = ax.get_figure()
    width, height = map(int, fig.get_size_inches() * fig.dpi)
    position = ax.get_position()

    annotationsHeight = 0.08
    new_y = position.y1 - position.height * (annotationsHeight)
    ax.set_position([position.x0,
                    position.y0,
                    position.width,
                    new_y - position.y0], 'original')

    zero_display = ax.transData.transform_point((0, 0))
    zero_figure = fig.transFigure.inverted().transform_point(zero_display)

    x_points = ax.lines[0].get_xdata()
    max_x_display = ax.transData.transform_point((len(x_points), 0))
    max_x_figure = fig.transFigure.inverted().transform_point(max_x_display)

    imgHeight = int(height * annotationsHeight)
    imgShape = image.imread(input_sequence[0]).shape
    imgAspectRatio = imgShape[1] / imgShape[0]
    imgWidth = int(imgAspectRatio * imgHeight)
    spaceSize = 0.07
    spacedImageSize = int(imgWidth * (1.0 + spaceSize))
    nImages = int((max_x_figure[0] - zero_figure[0])
            * width) // spacedImageSize

    extremums = find_interval_extremums(vals, nImages, binop)
    extr_indexes = [extr[0] for extr in extremums]

    for extr in extremums[:-1]:
        ax.plot(extr[0], extr[1], '*r', markersize=10)

    for it in range(nImages - 1):
        imgIdx = extr_indexes[it]
        lineIdx = len(input_sequence) // nImages * it
        img = image.imread(input_sequence[imgIdx])
        img = cv2.resize(img, dsize=(imgWidth, imgHeight))
        imgPoint = ax.transData.transform_point((lineIdx, 0))
        figPoint = fig.transFigure.inverted().transform_point((imgPoint[0]
                + spaceSize * imgWidth // 2, imgHeight))
        frac_to_pix = (fig.get_size_inches() * fig.dpi)

        ax.get_figure().figimage(img, figPoint[0] * frac_to_pix[0],
                height - figPoint[1] * frac_to_pix[1])

    return mlines.Line2D([], [], color='red', marker='*',
            markersize=10, label='Local extremum', linestyle='None')


def add_images_under(ax, input_sequence):
    """adds images under the graph fixed
    distance apart from each other"""
    fig = ax.get_figure()
    width, height = map(int, fig.get_size_inches() * fig.dpi)
    position = ax.get_position()

    annotationsHeight = 0.04
    new_y = position.y0 + position.height * (annotationsHeight)
    ax.set_position([position.x0,
                    new_y,
                    position.width,
                    position.y1 - new_y], 'original')

    zero_display = ax.transData.transform_point((0, 0))
    zero_figure = fig.transFigure.inverted().transform_point(zero_display)

    x_points = ax.lines[0].get_xdata()
    max_x_display = ax.transData.transform_point((len(x_points), 0))
    max_x_figure = fig.transFigure.inverted().transform_point(max_x_display)

    imgHeight = int(height * annotationsHeight)
    imgShape = image.imread(input_sequence[0]).shape
    imgAspectRatio = imgShape[1] / imgShape[0]
    imgWidth = int(imgAspectRatio * imgHeight)
    spaceSize = 0.07
    spacedImageSize = int(imgWidth * (1.0 + spaceSize))

    nImages = int((max_x_figure[0] - zero_figure[0])
            * width) // spacedImageSize

    for it in range(nImages):
        imgIdx = len(input_sequence) // nImages * it
        ax.axvline(x=imgIdx, linewidth=0.25
                + (it % 2 == 0) * 0.20, color='red')
        img = image.imread(input_sequence[imgIdx])
        img = cv2.resize(img, dsize=(imgWidth, imgHeight))
        imgPoint = ax.transData.transform_point((imgIdx, 0))
        figPoint = fig.transFigure.inverted().transform_point(
                (imgPoint[0] - imgWidth // 2, 0))
        frac_to_pix = (fig.get_size_inches() * fig.dpi)
        ax.get_figure().figimage(img, figPoint[0] * frac_to_pix[0],
                figPoint[1] * frac_to_pix[1])


def ious_combined_graph(test_results, input_sequence):
    """generation of graph of ious from all passes
    """
    plt.figure()
    ious = pd.concat([test['iou'] for test in test_results],
            axis=1, join='inner')
    mean = ious.mean(axis=1)
    colors = cm.get_cmap('Pastel1')(np.linspace(0.0, 1.0, len(test_results)))
    ax = ious.plot(legend=False, grid=False, color=colors,
            title='Accuracy for'
            + ' every pass and their mean. Images above are'
            + ' minimums on interval noted by red asterisks on the graph.'
            + ' Images below are frames marked by red vertical'
            + ' lines.')

    fig = ax.get_figure()
    fig.subplots_adjust(left=0.03, right=1.0, bottom=0.08, top=0.93)

    for line in ax.lines:
        line.set_linewidth(1)
    # smoothed = [max(0, x) for x in savgol_filter(mean, 15, 3)]
    smoothed = mean
    ax.plot(smoothed, color='#0D0080', linewidth=1)
    ax.set_xlabel('Frame number')
    ax.set_ylabel('IoU')
    ax.grid(color='white', lw=0.75)

    add_images_under(ax, input_sequence)
    extr_legend = add_extremums_above(ax, smoothed, input_sequence, min)

    ax.legend(handles=[mlines.Line2D([], [], color='#0D0080',
                          markersize=15, label='Mean'),
                          mlines.Line2D([], [], color='#C0C0C0',
                          markersize=15, label='Raw'),
                          extr_legend
                          ])

    return {'ious_combined.png': fig}


def ious_stdev_graph(test_results, input_sequence):
    """mean and its standard deviation at each point"""
    plt.figure()
    ious = pd.concat([test['iou'] for test in test_results],
            axis=1, join='inner')
    ious = ious.transpose()
    ious_mean = ious.mean()
    ax = ious_mean.plot(legend=False, grid=True, title='IoU mean')
    fig = ax.get_figure()
    fig.subplots_adjust(left=0.03, right=1.0, bottom=0.08, top=0.93)

    return {'ious_stdev.png': fig}


def generate(name, test, link, test_results, tests_input_path, stopwatch_test):
    env = Environment(
        loader=FileSystemLoader(searchpath="template"))
    template = env.get_template("policy_test.html")

    input_sequence_path = tests_input_path / test
    imgs = [f for f in input_sequence_path.glob('*') if f.suffix != '.ann']
    seq_paths = {int(p.stem) - 1: p for p in imgs}

    stopwatch_table = generate_statistics(
            {name: sum([durations_to_fps(timepoints_to_durations(p[name])) \
                    for p in stopwatch_test],
                    []) for name in stopwatch_test[0]})
    iou_table = generate_statistics({'iou': sum([[v for v in dt['iou']] \
            for dt in test_results], [])})

    images = {}
    images.update(ious_combined_graph(test_results, seq_paths))
    images.update(ious_stdev_graph(test_results, seq_paths))
    generate_images(images, 'output' / link / 'images')

    summary_path = 'output' / link / 'index.html'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(template.render(
            policyname=name,
            testname=test,
            link=link,
            test_results=test_results,
            stopwatch_table=stopwatch_table,
            iou_table=iou_table))
