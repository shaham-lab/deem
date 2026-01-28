#! /usr/bin/env python

"""
Copyright (c) 2018 Vaibhav B Sinha, Sukrut Rao, Vineeth N Balasubramanian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import os
import sys
import time
import numpy as np

def compute_mean_std(accuracies,verbose=True):
    """
    Computes the mean and standard deviation of a list of accuracies.
    
    Args:
        accuracies (list): A list of accuracy values.
    
    Returns:
        tuple: A tuple containing the mean and standard deviation of the accuracies.
    """
    mean = np.mean(accuracies) 
    std = np.std(accuracies)
    if verbose:
        print('^Accs^')
        print(f"Mean accuracy: {mean:.4f} ± {std:.4f}")
        print(f"Mean accuracy * 100: {mean * 100:.4f} ± {std * 100:.4f}")
    return mean, std

def main():
    accuracies = []
    parser = argparse.ArgumentParser(
        description='Run the Dawid-Skene, Fast Dawid-Skene, the Hybrid, or the Majority Voting Algorithm')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to use')
    parser.add_argument('--k', default=0, type=int, required=False,
                        help='Number of annotators to use. Each data point must have at least K annotators. If more annotators are available, the first K annotators are used. If K = 0, then all available annotations for each data point are used. Default is 0')
    parser.add_argument('--algorithm', type=str, choices=['DS', 'FDS', 'H', 'MV'], required=True,
                        help='Algorithm to use - DS: Dawid-Skene, FDS: Fast-Dawid Skene, H: Hybrid, MV: Majority Voting')
    parser.add_argument('--mode', default='aggregate', type=str, choices=[
                        'aggregate', 'test'], required=False, help='The mode to run this program - aggregate: obtain aggregated dataset, test: aggregate data and compare with ground truths. Default is aggregate')
    parser.add_argument('--crowd_annotations_path', default=None, type=str, required=False,
                        help='Path to crowdsourced annotations. Default is crowd.csv inside the dataset directory')
    parser.add_argument('--ground_truths_path', default=None, type=str, required=False,
                        help='Path to ground truths, if using test mode. Default is gold.csv inside the dataset directory')
    parser.add_argument('--dataset_path', default=None, type=str,
                        required=False, help='Custom path to dataset, to override default')
    parser.add_argument('--seed', default=18, type=int,
                        required=False, help='Sets the random seed. Default is 18')
    parser.add_argument('--output', default=None, type=str, required=False,
                        help='Path to write CSV output, output is not written if this is not set')
    parser.add_argument('--print_result', action='store_true',
                        help='Prints the predictions and accuracy to standard output, if set')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Run in verbose mode', dest='verbose')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Run 5 seeds', dest='a', default=False)
    parser.add_argument('-s', '--save', action='store_true',
                        help='Saves predictions', dest='save', default=False)
    args = parser.parse_args()

    times = []
    if args.a:
        for i in range (0,5):
            args.seed = i
            np.random.seed(args.seed) #:.2f
            start_time = time.perf_counter()
            accuracies.append(run(args))
            times.append(time.perf_counter() - start_time)
    else:
        np.random.seed(args.seed)
        start_time = time.perf_counter()
        accuracies.append(run(args))
        times.append(time.perf_counter() - start_time)

    compute_mean_std(accuracies=accuracies)
    print(f'Average time: {np.mean(times):.2f} seconds, for {len(accuracies)} seeds')

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, '..'))
    from fast_dawid_skene.main import run
    main()


# Example python scripts/fast_dawid_skene.py --dataset mixedclf_test --mode test --algorithm H --seed 0 -a   