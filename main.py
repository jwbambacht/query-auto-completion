import pickle
import pyltr
from data import Data
import numpy as np
import os

def train_model(n, overwrite=False,withFeatures=True):
    if withFeatures:
        modelName = "data/model-withFeatures.sav"
    else:
        modelName = "data/model-withoutFeatures.sav"

    if overwrite or not os.path.isfile(modelName):
        dataset = Data()
        training_data = dataset.get_training_samples(n)
        training_data = sorted(training_data, key=lambda x: x[0])
        qid = [x[0] for x in training_data]
        y = [x[len(training_data[0]) - 1] for x in training_data]

        if withFeatures:
            x = [x[1:len(training_data[0]) - 1] for x in training_data]
        else:
            x = [x[7:len(training_data[0]) - 1] for x in training_data]

        LM = pyltr.models.LambdaMART(n_estimators=300)
        LM.fit(x, y, qid)
        print("Done training")

        pickle.dump(LM, open(modelName, 'wb'))
        print("Model written to file")
        return LM
    else:
        LM = pickle.load(open(modelName, "rb"))
        print("Model loaded from file")
        return LM


def calculate_mrr_lm(prefix_count, popular_prefix_count, n, withFeatures=True):
    lm = train_model(n, False, withFeatures)

    dataset = Data()
    testing_data = dataset.get_testing_samples(10000)

    ranks_divided = {"Frequent": [], "Rare": [], "Unseen": []}
    prefixes = dataset.get_testing_prefixes(10000)

    i = 0
    ranks = []
    while i < len(testing_data):
        curr_id = testing_data[i][0]
        j = i + 1
        while j < len(testing_data) and curr_id == testing_data[j][0]:
            j += 1

        pred = lm.predict([x[:len(testing_data[0]) - 1] for x in testing_data[i:j]])
        pred = zip(pred, [y[len(testing_data[0]) - 1] for y in testing_data[i:j]])
        pred = sorted(pred, reverse=True)

        index = list(map(lambda x: x[1], pred)).index(1)
        count = np.sum(list(map(lambda x: x[0] == pred[index][0], pred)))
        avg_rank = 0
        for k in range(count):
            avg_rank += index + 1 + k
        avg_rank = avg_rank / count

        if len(pred) == 10:
            prefix = prefixes[i]
            ranks_divided[dataset.get_prefix_popularity(prefix, prefix_count, popular_prefix_count)] += [avg_rank]
            ranks.append(avg_rank)
        i = j

    print("LM (features={}) lengths: {}, {}, {}".format(len(ranks_divided["Frequent"]), len(ranks_divided["Rare"]),
                                           len(ranks_divided["Unseen"])))

    mrr_divided = {t: np.mean(list(map(lambda x: 1 / x if x <= 8 else 0, ranking))) for t, ranking in
                   ranks_divided.items()}
    mrr = np.mean(list(map(lambda x: 1 / x if x <= 8 else 0, ranks)))
    return mrr, mrr_divided


def calculate_mrr_mpc(prefix_count, popular_prefix_count):
    dataset = Data()
    testing_data = dataset.get_testing_samples(10000)

    ranks_divided = {"Frequent": [], "Rare": [], "Unseen": []}
    prefixes = dataset.get_testing_prefixes(10000)

    i = 0
    ranks = []
    while i < len(testing_data):
        curr_id = testing_data[i][0]
        j = i + 1
        while j < len(testing_data) and curr_id == testing_data[j][0]:
            j += 1

        pred = [x[7] for x in testing_data[i:j]]
        pred = zip(pred, [y[len(testing_data[0]) - 1] for y in testing_data[i:j]])
        pred = sorted(pred, reverse=True)

        index = list(map(lambda x: x[1], pred)).index(1)
        count = np.sum(list(map(lambda x: x[0] == pred[index][0], pred)))
        avg_rank = 0
        if pred[index][0] != 0:
            for k in range(count):
                avg_rank += index + 1 + k
            avg_rank = avg_rank / count
        else:
            avg_rank = 10

        if len(pred) == 10:
            prefix = prefixes[i]
            ranks_divided[dataset.get_prefix_popularity(prefix, prefix_count, popular_prefix_count)] += [avg_rank]
            ranks.append(avg_rank)
        i = j


    print("MPC lengths: {}, {}, {}".format(len(ranks_divided["Frequent"]), len(ranks_divided["Rare"]), len(ranks_divided["Unseen"])))

    mrr_divided = {t: np.mean(list(map(lambda x: 1 / x if x <= 8 else 0, ranking))) for t, ranking in ranks_divided.items()}
    mrr = np.mean(list(map(lambda x: 1 / x if x <= 8 else 0, ranks)))
    return mrr, mrr_divided


def main():
    data = Data()
    prefixes = data.get_prefix_counts(overwrite=False)
    popular_prefix_count = sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[100000][1]

    n = 10000

    mrr_lm_with_features = calculate_mrr_lm(prefixes, popular_prefix_count, n, withFeatures=True)
    mrr_lm_without_features = calculate_mrr_lm(prefixes, popular_prefix_count, n, withFeatures=False)
    mrr_mpc = calculate_mrr_mpc(prefixes, popular_prefix_count)
    
    print("MRR LM with features: {}".format(mrr_lm_with_features))
    print("MRR LM without features: {}".format(mrr_lm_without_features))
    print("MRR MPC: {}".format(mrr_mpc))


main()
