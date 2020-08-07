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

env = Environment(
    loader=FileSystemLoader(searchpath="template")
)

def read_dir(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    csvs = []
    stopwatch = []
    for file in files:
        suffix = "_stopwatch.csv"
        full_path = Path(path) / Path(file)
        if len(file) > len(suffix) and file[-len(suffix):] == suffix:
            stopwatch.append(full_path)
        elif Path(file).suffix == '.csv':
            csvs.append(full_path)
    return csvs, read_csvs(csvs), [read_stopwatch(file) for file in stopwatch]

def read_csvs(filenames):
    tables = []
    for filename in filenames:
        with open(filename) as file:
            csv_reader = csv.reader(file, delimiter=',')
            table = []
            numeration = {}
            for line_count, row in enumerate(csv_reader):
                if line_count > 0:
                    table.append({})
                for column_count, field in enumerate(row):
                    if line_count == 0:
                        numeration[column_count] = field
                    else:
                        table[line_count - 1][numeration[column_count]] = field
            if len(table):
                tables.append(table)
    return tables

def read_stopwatch(filename):
    stopwatch_stats = []
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        stopwatch_stats = []
        for row in csv_reader:
            raw_data = list(map(float, row[1:]))
            if len(raw_data) == 0:
                continue
            data = []
            for idx in range(len(raw_data) // 2):
                data.append(1.0 / (raw_data[idx * 2 + 1] - raw_data[idx * 2]))
            stats = make_stopwatch_stats(data)
            stats['name'] = row[0]
            stopwatch_stats.append(stats)   
    return stopwatch_stats

def make_stopwatch_stats(data):
    
    stats = {}
    stats['mean'] = statistics.mean(data)
    stats['median'] = statistics.median(data)
    stats['stdev'] = statistics.stdev(data)
    stats['data'] = data
    return stats

def create_title(config, nTests):
    title = ""
    with open(config) as cfg:
        visualize, fps, policy = cfg.readline().split()
        title = f"{policy} running at {'unlimited' if int(fps) == 0 else f'forced {fps}'} fps on {nTests} {'tests' if nTests != 1 else 'test'}"
    return title

def save_and_clear_plot(filename):
    plt.savefig(Path('images') / Path(filename))
    plt.cla()

def plot_iou_percents(lst, filename):
    percents = [0] * 101
    for entry in lst:
        percents[int(entry * 100)] += 1

    percents = list(itertools.accumulate(percents))
    plt.plot(percents)

    plt.xlabel('IoU')
    plt.ylabel('number of frames with this score or lower')
    save_and_clear_plot(filename)

def bar_iou_percents(lst, filename):
    perdimes = [0] * 11
    for entry in lst:
        perdimes[int(entry * 10)] += 1

    plt.bar(range(11), perdimes)

    plt.xlabel('IoU')
    plt.ylabel('number of frames with this score (scale 0 - 10)')
    save_and_clear_plot(filename)

def bar_iou_per_test(lst, names, filename):
    plt.bar([str(name.stem) for name in names], lst)
    plt.xlabel('tests')
    plt.ylabel('average IoU')
    save_and_clear_plot(filename)

@dataclass
class SingleTestStats:
    name: str = ""
    mean_iou: float = 0.0
    median_iou: float = 0.0
    stdev_iou: float = 0.0
    nFrames: int = 0
    iou_frame_filename: str = ""
    iou_bbox_size_filename: str = ""

def plot_iou_frame(ious, filename):
    plt.plot(ious, color='#CACACA')
    smoothed = [max(0, x) for x in savgol_filter(ious, len(ious) // 30 * 2 + 1, 2)]
    plt.plot(smoothed, linewidth=2)
    plt.xlabel('Frame number')
    plt.ylabel('IoU')
    save_and_clear_plot(filename)

@dataclass
class Bbox:
    left: int
    top: int
    width: int
    height: int

    def size(self):
        return self.width * self.height

def get_bboxes(table):
    return [Bbox(int(row['left']), int(row['top']), int(row['width']), int(row['height']))
            for row in table]

def get_real_bboxes(table):
    return [Bbox(int(row['realLeft']), int(row['realTop']), int(row['realWidth']),
        int(row['realHeight'])) for row in table]

def plot_iou_bbox_size(table, filename):
    bboxes = get_real_bboxes(table)
    max_size = max([bbox.size() for bbox in bboxes])
    min_size = min([bbox.size() for bbox in bboxes])
    scale_down_factor = max(1, max_size // 100)
    size = (max_size) // scale_down_factor + 1
    iou_sum = size * [0]
    count = size * [0]
    for counter, bbox in enumerate(bboxes):
        iou = float(table[counter]['iou'])
        iou_sum[bbox.size() // scale_down_factor] += iou
        count[bbox.size() // scale_down_factor] += 1
    average = [s / max(1, count[index]) for index, s in enumerate(iou_sum)]
    smoothed = [max(0, x) for x in savgol_filter(average, len(average) // 10 * 2 + 1, 2)]
    real_size = [math.sqrt(elem * scale_down_factor) for elem in range(size)]
    plt.plot(real_size[min_size // scale_down_factor:],
            smoothed[min_size // scale_down_factor:])
    plt.xlabel('Real bounding box side length')
    plt.ylabel('Avarage IoU')
    save_and_clear_plot(filename)

def create_html(tables, config, filenames, stopwatch):
    # aggregated from all tests
    title = create_title(config, len(tables))
    aggregated_ious = [float(row['iou']) for table in tables for row in table]
    mean_iou = statistics.mean(aggregated_ious)
    median_iou = statistics.median(aggregated_ious)
    stdev_iou = statistics.stdev(aggregated_ious)
    iou_percents_aggregated_filename = 'iou_percents_aggregated.png'
    plot_iou_percents(aggregated_ious, iou_percents_aggregated_filename)
    bar_percents_aggregated_filename = 'bar_percents_aggregated.png'
    bar_iou_percents(aggregated_ious, bar_percents_aggregated_filename)
    
    # comparison between tests
    iou_per_test_filename = 'iou_per_test.png'
    bar_iou_per_test(list(map(statistics.mean, [[float(row['iou']) for row in table] \
            for table in tables])),
            filenames, iou_per_test_filename)

    # statistics for every test
    separate_tests_stats = []
    for counter, table in enumerate(tables):
        test = SingleTestStats()
        test.name = Path(filenames[counter]).stem
        ious = [float(row['iou']) for row in table]
        test.mean_iou = statistics.mean(ious)
        test.median_iou = statistics.median(ious)
        test.stdev_iou = statistics.stdev(ious)
        test.nFrames = len(table)
        test.iou_frame_filename = test.name + '_iou_frame.png'
        plot_iou_frame(ious, test.iou_frame_filename)
        test.iou_bbox_size_filename = test.name + '_iou_bbox_size.png'
        plot_iou_bbox_size(table, test.iou_bbox_size_filename)

        separate_tests_stats.append(test)

    # stopwatch data
    
    stopwatch_data_aggregated = defaultdict(list)
    for tests in stopwatch:
        for algs in tests:
            stopwatch_data_aggregated[algs['name']] += algs['data']
    stopwatch_stats_aggregated = [{**make_stopwatch_stats(value), **{'name': key}} \
            for key, value in stopwatch_data_aggregated.items()]

    # create the html
    template = env.get_template("report.html")
    with open("output/report.html", 'w') as f:
        f.write(template.render(
            title=title,
            mean_iou=mean_iou,
            median_iou=median_iou,
            stdev_iou=stdev_iou,
            iou_percents_aggregated_filename=iou_percents_aggregated_filename,
            bar_percents_aggregated_filename=bar_percents_aggregated_filename,
            iou_per_test_filename=iou_per_test_filename,
            separate_tests_stats=separate_tests_stats,
            stopwatch_stats_aggregated=stopwatch_stats_aggregated
            ))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    filenames, csvs, stopwatch = read_dir(args.input_dir)
    create_html(csvs, args.config, filenames, stopwatch)

if __name__ == "__main__":
    main()
