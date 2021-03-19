#!/usr/bin/env python3

import vot_report_generator.index as index
import vot_report_generator.policy_index as  policy_index
import vot_report_generator.policy_summary as  policy_summary
import vot_report_generator.policy_test as  policy_test

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def readidx(path):
    index = {}
    with open(path, 'r') as file:
        for row in file:
            key, value = row.split('=')
            value = value[:-1]
            index[key] = value
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'tests_output_path',
        help='Path to the tester node results',
        type=Path
    )
    parser.add_argument(
        'tests_input_path',
        help='Path to the directory with input video sequences in ALOV format',
        type=Path
    )
    parser.add_argument(
        '--output',
        help='Path to the output directory',
        type=Path,
        default=Path('output')
    )
    args = parser.parse_args()
    idx = readidx(args.tests_output_path / 'test.index')

    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (21, 7)
    plt.rcParams['figure.max_open_warning'] = 0

    policy_links = {}
    policy_fps = {}
    test_results = {}
    stopwatch_results = {}
    for policyit in range(int(idx['policies'])):
        name = idx[f'policyname{policyit + 1}']
        fps = int(idx[f'policyfps{policyit + 1}'])
        policy_links[name] = Path(name)
        policy_fps[name] = fps
        test_results[name] = {}
        stopwatch_results[name] = {}
        for testit in range(int(idx['tests'])):
            testname = idx[f'test{testit + 1}']
            passes = int(idx['passes'])
            test_results[name][testname] = []
            stopwatch_results[name][testname] = []
            for passit in range(passes):
                test_results[name][testname].append(
                    pd.read_csv(args.tests_output_path
                                / name
                                / f'{testname}_{passit + 1}.csv'))
                with open(args.tests_output_path
                          / name
                          / f'{testname}_{passit + 1}_stopwatch.csv') \
                        as f:
                    cols = {}
                    for line in f:
                        li = line[:-1].split(',')
                        cols[li[0]] = []
                        for entry in li[1:]:
                            cols[li[0]].append(float(entry))
                    stopwatch_results[name][testname].append(cols)

    index.generate(idx, policy_links, args.output)

    for name, link in policy_links.items():
        fps = policy_fps[name]
        policy_index.generate(
            name,
            link,
            fps,
            idx,
            test_results[name],
            args.output
        )
        policy_summary.generate(name, link, fps, idx, test_results[name],
                                stopwatch_results[name], args.output)
        for test in test_results[name]:
            policy_test.generate(name, test, link / test,
                                 test_results[name][test],
                                 args.tests_input_path,
                                 stopwatch_results[name][test],
                                 args.output)


if __name__ == "__main__":
    main()
