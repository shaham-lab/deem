import argparse
import random
import time
from results_summary import OfflineResultsSummary
from data_pipeline import gete2wlandw2el, getaccuracy
from LA_methods.baselines.c_EM import EM


def from_z_to_truths_multi(z_i,label_set):
    truths = {}
    for item in z_i.keys():
        votes = z_i.get(item)
        candidate = []
        max_ = -1
        for class_ in label_set:
            if votes.get(class_) is None:
                continue
            if votes.get(class_) > max_:
                candidate.clear()
                candidate.append(class_)
                max_ = votes.get(class_)
            elif votes.get(class_) == max_:
                candidate.append(class_)
        truths[item] = random.choice(candidate)
    return truths

if __name__ == "__main__":

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
        ('mill', 'mill'),

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
    
    results = OfflineResultsSummary('DS')
    num_round = 20
    seed = 1
    random.seed(seed)
    for dataset, abbrev in datasets:

        label_path = "data/" + dataset + "/label.csv"
        truth_path = "data/" + dataset + "/truth.csv"
        random.seed(seed)
        list_ = []

        e2wl, w2el, label_set = gete2wlandw2el(label_path)

        for round in range(num_round):
            print(abbrev, 'round', round + 1)
            t1 = time.time()
            e2lpd, w2cm = EM(e2wl, w2el, label_set).Run()
            truths = from_z_to_truths_multi(e2lpd,label_set)
            t2 = time.time()
            acc = getaccuracy(truth_path, truths)
            results.add(abbrev, acc, t2 - t1, 20)

    print('DS results')
    print(results.to_dataframe_mean())