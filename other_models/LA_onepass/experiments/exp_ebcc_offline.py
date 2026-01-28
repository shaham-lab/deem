import argparse
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import gete2wlandw2el,getaccuracy
import time
from results_summary import OfflineResultsSummary
from LA_methods.baselines.ebcc import ebcc_vb

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
        ('deem/hs3_train_dataset','HS3'),
        ('deem/mniste_train_dataset','MnistE'),
        ('deem/petfinder_train_dataset','Petfinder'),
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
    
    results = OfflineResultsSummary('EBCC')
    num_round = 10
    seed = 1
    for dataset, abbrev in datasets:
        print(dataset, abbrev)
        label_path = "data/"+dataset+"/label.csv"
        truth_path = "data/"+dataset+"/truth.csv"
        random.seed(seed)
        list_ = []

        e2wl, w2el, label_set = gete2wlandw2el(label_path)

        for round in range(num_round):
            print('dataset',abbrev,'round',round+1)
            t1 = time.time()
            truths,elbo, iterations = ebcc_vb(e2wl,w2el,label_set)
            t2 = time.time()
            acc = getaccuracy(truth_path, truths)
            print(f'dataset {abbrev} acc: {acc}')
            results.add(abbrev,acc,t2-t1,iterations,elbo)

    print('EBCC results')
    print(results.get_EBCC_dataframe())
