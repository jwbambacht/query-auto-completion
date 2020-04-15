import pickle
import pyltr
from dataset import Dataset
from data import Data
import numpy as np
import os

def train_model(overwrite=False,withFeatures=True):
    if withFeatures:
        modelName = "data/model-withFeatures.sav"
    else:
        modelName = "data/model-withoutFeatures.sav"

    if overwrite or not os.path.isfile(modelName):
        dataset = Data()
        training_data = dataset.get_training_samples(0)
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


def calculate_mrr_lm(withFeatures=True):
    lm = train_model(False,withFeatures)

    dataset = Data()
    testing_data = dataset.get_testing_samples(0)

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
            ranks.append(avg_rank)
        i = j

    mrr = np.mean(list(map(lambda x: 1 / x if x <= 8 else 0, ranks)))
    return mrr


def calculate_mrr_mpc():
    dataset = Data()
    testing_data = dataset.get_testing_samples(0)

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
            print(avg_rank)
            ranks.append(avg_rank)
        i = j

    mrr = np.mean(list(map(lambda x: 1 / x if x <= 8 else 0, ranks)))
    return mrr


def main():
    mrr_lm_with_features = calculate_mrr_lm(True)
    mrr_lm_without_features = calculate_mrr_lm(False)
    mrr_mpc = calculate_mrr_mpc()
    
    print("MRR LM with features: {}".format(mrr_lm_with_features))
    print("MRR LM without features: {}".format(mrr_lm_without_features))
    print("MRR MPC: {}".format(mrr_mpc))


# main()
data = Data()
data.get_training_samples(0, overwrite=True)
