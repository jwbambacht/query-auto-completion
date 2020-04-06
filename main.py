import pickle
import pyltr
import sklearn
from dataset import Dataset
import numpy as np
import os


def train_model(overwrite=False):
    if overwrite or not os.path.isfile("data/model.sav"):
        dataset = Dataset()
        training_data = dataset.get_training_samples()
        training_data = sorted(training_data, key=lambda x: x[0])
        qid = [x[0] for x in training_data]
        y = [x[len(training_data[0]) - 1] for x in training_data]
        x = [x[1:len(training_data[0]) - 1] for x in training_data]

        LM = pyltr.models.LambdaMART(n_estimators=300)
        LM.fit(x, y, qid)
        print("Done training")

        pickle.dump(LM, open("data/model.sav", 'wb'))
        print("Model written to file")
        return LM
    else:
        LM = pickle.load(open("data/model.sav", "rb"))
        print("Model loaded from file")
        return LM


def main():
    LM = train_model(False)

    dataset = Dataset()
    testing_data = dataset.get_testing_samples()

    i = 0
    ranks = []
    while i < len(testing_data):
        curr_id = testing_data[i][1]
        j = i + 1
        while j < len(testing_data) and curr_id == testing_data[j][1]:
            j += 1

        pred = LM.predict([x[2:len(testing_data[0]) - 1] for x in testing_data[i:j]])
        pred = zip(pred, [y[len(testing_data[0]) - 1] for y in testing_data[i:j]])
        pred = sorted(pred, reverse=True)

        index = map(lambda x: x[1], pred).index(1)
        count = np.sum(map(lambda x: x == pred[index][0], pred))
        avg_rank = 0
        for k in range(count):
            avg_rank += index + 1 + k
        avg_rank = avg_rank/count
        ranks.append(avg_rank)
        i = j

    print(ranks)
    MRR = np.mean(map(lambda x: 1/x, ranks))
    print(MRR)


main()

