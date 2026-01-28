import random
import sys
import os

import argparse


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import gete2wlandw2el,getaccuracy, save_preds
import time
from results_summary import OfflineResultsSummary
from LA_methods.proposed.proposed_method import one_pass,two_pass
random.seed(1)

if __name__ == '__main__':


    datasets = [
        ('crowdscale2013/sentiment', 'senti'),
        ('crowdscale2013/fact_eval', 'fact'),
        ('active-crowd-toolkit/CF', 'CF'),
        ('active-crowd-toolkit/CF_amt', 'CF_amt'),
        ('active-crowd-toolkit/MS', 'MS'),
        ('crowd_truth_inference/s4_Face Sentiment Identification', 'face'),
        ('crowd_truth_inference/s5_AdultContent', 'adult'),
        ('SpectralMethodsMeetEM/dog', 'dog'),
        ('SpectralMethodsMeetEM/web', 'web'),
        ('mill','mill'),

        ('active-crowd-toolkit/SP', 'SP'),
        ('active-crowd-toolkit/SP_amt', 'SP_amt'),
        ('active-crowd-toolkit/ZenCrowd_all', 'ZC_all'),
        ('active-crowd-toolkit/ZenCrowd_in', 'ZC_in'),
        ('active-crowd-toolkit/ZenCrowd_us', 'ZC_us'),
        ('crowd_truth_inference/d_jn-product', 'product'),
        ('crowd_truth_inference/d_sentiment', 'tweet'),
        ('SpectralMethodsMeetEM/bluebird', 'bird'),
        ('SpectralMethodsMeetEM/rte', 'rte'),
        ('SpectralMethodsMeetEM/trec', 'trec'),
    ]
    
    datasets = [
        ('rbm/mniste_cs_test_dataset','MnistE'),
        ('rbm/condind_test_dataset','CondInd'),
        ('rbm/hs3_test_dataset','HS3'),
        ('rbm/mixedclf_test_dataset','MixedClf'),
        ('rbm/nnt_test_dataset','NeuralNet'),
        ('rbm/c2_boosting_test_dataset','C-Boosting'),
        ('rbm/c2_complement_test_dataset','C-Complement'),
        
    ]
    datasets = [
        #('deem/mniste_train_dataset','MnistE'),
        #('deem/hs3_train_dataset','HS3'),
        #('deem/petfinder_train_dataset','Petfinder'),
        ('deem/mboosting_train_dataset','MBoosting'),
        ('deem/mcomplement_train_dataset','MComplement'),
    ]
    datasets = [
    #('deem/hs3_train_dataset', 'HS3'),
    #('deem/mniste_train_dataset', 'MnistE'),
    #('deem/petfinder_train_dataset', 'Petfinder'),
    #('deem/artificial-characters_train_dataset', 'ArtificialCharacters'),
    #('deem/csgo_train_dataset', 'CSGO'),
    #('deem/eye_movements_train_dataset', 'EyeMovements'),
    #('deem/GesturePhaseSegmentationProcessed_train_dataset', 'GesturePhaseSegmentation'),
    #('deem/mboosting_train_dataset', 'MBoosting'),
    #('deem/mcomplement_train_dataset', 'MComplement'),
    #('deem/microaggregation2_train_dataset', 'Microaggregation2'),
    #('deem/pendigits_train_dataset', 'Pendigits'),
    #('deem/volcanoes-b2_train_dataset', 'VolcanoesB2'),
    ('deem/tree3k_train_dataset', 'Tree3k')
    ]

    parser = argparse.ArgumentParser(description='Offline Results Summary')
    parser.add_argument('--datasets', type=str, default='all')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Saves predictions', dest='save', default=False)
    parser.add_argument('--seeds', default=10, type=int,
                        required=False, help='Sets how many seeds to run, defaults 10')
    args = parser.parse_args()

    if args.datasets == 'all':
        datasets = datasets
    else:
        datasets = [(p, p.split('/')[-1].replace('_train_dataset', ''))
                    for p in args.datasets.split(',')
                    ]
    

    results_onepass = OfflineResultsSummary('onepass')
    results_twopass = OfflineResultsSummary('twopass')

    from pathlib import Path

    scripts_folder = Path(__file__).parent.resolve()
    data_folder = scripts_folder.parent / 'data'

    round = args.seeds
    for dataset, abbrev in datasets:
        print(dataset, abbrev)
        label_path = f"{data_folder}/"+dataset+"/label.csv"
        truth_path = f"{data_folder}/"+dataset+"/truth.csv"

        e2wl, w2el, label_set = gete2wlandw2el(label_path)
        for r in range(round):
            t1 = time.time()
            one_pass_truths, a = one_pass(e2wl, w2el, label_set, alpha=2, beta=2)
            t2 = time.time()

            one_pass_acc = getaccuracy(truth_path, one_pass_truths)


            t3 = time.time()
            two_pass_truths = two_pass(e2wl, a, label_set)
            t4 = time.time()
            two_pass_acc = getaccuracy(truth_path, two_pass_truths)

            if args.save:
                save_preds(truth_path, one_pass_truths,r)
                save_preds(truth_path, two_pass_truths,r,'two')

            results_onepass.add(abbrev, one_pass_acc, t2 - t1, 1)
            results_twopass.add(abbrev, two_pass_acc, t2 - t1 + t4 - t3, 2)

            
    import pandas as pd
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Let pandas choose best width
    pd.set_option('display.max_colwidth', None)  # Show full content in each column (for older pandas)
    pd.set_option('display.expand_frame_repr', False)  # Avoid wrapping in smaller screens
    print('One pass:\n',results_onepass.to_dataframe_mean())
    print('Two pass:\n',results_twopass.to_dataframe_mean())


# python experiments/exp_proposed_offline.py