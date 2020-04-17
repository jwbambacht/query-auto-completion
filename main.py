import pickle
import pyltr
from data import Data
import numpy as np
import os
from matplotlib import pyplot as plt

def train_model(overwrite=False,withFeatures=True,n=0):
    if withFeatures:
        modelName = "data/model-"+str(n)+"-withFeatures.sav"
    else:
        modelName = "data/model-"+str(n)+"-withoutFeatures.sav"

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

def calculate_mrr_lm(prefix_count, popular_prefix_count, n, overwriteModel=False, withFeatures=True):
    lm = train_model(overwriteModel, withFeatures,n)

    dataset = Data()
    testing_data = dataset.get_testing_samples(10000)

    if len(prefix_count) > 0:
        ranks_divided = {"Frequent": [], "Rare": [], "Unseen": []}
        prefixes = dataset.get_testing_prefixes(n)

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
            if len(prefix_count) > 0:
                prefix = prefixes[i]
                ranks_divided[dataset.get_prefix_popularity(prefix, prefix_count, popular_prefix_count)] += [avg_rank]
            ranks.append(avg_rank)
        i = j

    if len(prefix_count) > 0:
        print("LM lengths: {}, {}, {}".format(len(ranks_divided["Frequent"]), len(ranks_divided["Rare"]), len(ranks_divided["Unseen"])))

        mrr_divided = {t: np.mean(list(map(lambda x: 1 / x if x <= 8 else 0, ranking))) for t, ranking in ranks_divided.items()}
    else:
        mrr_divided = 0

    mrr = np.mean(list(map(lambda x: 1 / x if x <= 8 else 0, ranks)))
    return mrr, mrr_divided


def calculate_mrr_mpc(prefix_count, popular_prefix_count, n):
    dataset = Data()
    testing_data = dataset.get_testing_samples(n)

    if len(prefix_count) > 0:
        ranks_divided = {"Frequent": [], "Rare": [], "Unseen": []}
        prefixes = dataset.get_testing_prefixes(n)

    i = 0
    ranks = []
    candidates = []
    while i < len(testing_data):
        curr_id = testing_data[i][0]
        j = i + 1
        while j < len(testing_data) and curr_id == testing_data[j][0]:
            j += 1

        pred = [x[7] for x in testing_data[i:j]]
        pred = zip(pred, [y[len(testing_data[0]) - 1] for y in testing_data[i:j]])
        pred = sorted(pred, reverse=True)
        
        candidates.append(len(pred))

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
            if len(prefix_count) > 0:
                prefix = prefixes[i]
                ranks_divided[dataset.get_prefix_popularity(prefix, prefix_count, popular_prefix_count)] += [avg_rank]
            ranks.append(avg_rank)
        i = j
        
    candidates_sum = [0,0,0,0,0,0,0,0,0]
    candidates_lower = []    

    for a in candidates:
        if a <= 10:
            candidates_sum[int(a)-2] = candidates_sum[int(a)-2] + 1
        if a < 10:
            candidates_lower.append(a)
    
    for r in range(len(candidates_sum)):
        print(f"{r}: {candidates_sum[r]}")
                
    ranks_ratio = [i/len(candidates)*100 for i in candidates_sum]
    
    print(f"Total candidates: {len(candidates)}")
    print(f"Total candidates <10: {len(candidates_lower)}")
    print(f"Mean candidates <10: {np.mean(candidates_lower)}")
    print(f"Ratio candidates <10: {len(candidates_lower)/len(candidates)}")
	
    if len(prefix_count) > 0:
        print("MPC lengths: {}, {}, {}".format(len(ranks_divided["Frequent"]), len(ranks_divided["Rare"]), len(ranks_divided["Unseen"])))

        mrr_divided = {t: np.mean(list(map(lambda x: 1 / x if x <= 8 else 0, ranking))) for t, ranking in ranks_divided.items()}
    else:
        mrr_divided = 0
        
    mrr = np.mean(list(map(lambda x: 1 / x if x <= 8 else 0, ranks)))
    return mrr, mrr_divided, ranks_ratio


def main():
    calculate_prefix_count      = False
    experiments_one_enabled     = True
    experiments_two_enabled     = False
    test_MPC_models             = False
    test_LM_models              = True
    overwriteModel              = False
    suggested_candidates_plot   = False

    if suggested_candidates_plot:
        calculate_prefix_count = False
        test_MPC_models = True
        test_LM_models = False
        experiments_one_enabled= True
        experiments_two_enabled= True

    if calculate_prefix_count:
        data = Data()
        prefixes = data.get_prefix_counts(overwrite=False)
        popular_prefix_count = sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[100000][1]
    else:
        prefixes = {}
        popular_prefix_count    = 0
    
    ## Experiment 1
    n                           = 0
    if experiments_one_enabled:
        if test_MPC_models:
            mrr_mpc_exp1, 					mrr_mpc_divided_exp1, 					ranks_mpc_ratio_exp1      = calculate_mrr_mpc(prefixes, popular_prefix_count, n)

        if test_LM_models:
            mrr_lm_without_features_exp1, 	mrr_lm_divided_without_features_exp1                              = calculate_mrr_lm(prefixes, popular_prefix_count, n, overwriteModel, withFeatures=False)
            mrr_lm_with_features_exp1, 		mrr_lm_divided_with_features_exp1                                 = calculate_mrr_lm(prefixes, popular_prefix_count, n, overwriteModel, withFeatures=True)
    
        print("Experiment 1 results, full-query based candidates only:")
        if test_MPC_models:
            print("MRR MPC: {}".format(mrr_mpc_exp1))

        if test_LM_models:
            print("MRR LM w/o features: {}".format(mrr_lm_without_features_exp1))
            print("MRR LM w/ features {}".format(mrr_lm_with_features_exp1))

        if calculate_prefix_count:
            if test_MPC_models:
                print("MRR MPC Divided: {}".format(mrr_mpc_divided_exp1))

            if test_LM_models:
                print("MRR LM w/o features Divided: {}".format(mrr_lm_divided_without_features_exp1))
                print("MRR LM w/ features Divided: {}".format(mrr_lm_divided_with_features_exp1))
    
    ## Experiment 2
    n                           = 10000
    if experiments_two_enabled:
        if test_MPC_models:
            mrr_mpc_exp2, 					mrr_mpc_divided_exp2, 					ranks_mpc_ratio_exp2     = calculate_mrr_mpc(prefixes, popular_prefix_count, n)

        if test_LM_models:
            mrr_lm_without_features_exp2, 	mrr_lm_divided_without_features_exp2                             = calculate_mrr_lm(prefixes, popular_prefix_count, n, overwriteModel, withFeatures=False)
            mrr_lm_with_features_exp2, 		mrr_lm_divided_with_features_exp2                                = calculate_mrr_lm(prefixes, popular_prefix_count, n, overwriteModel, withFeatures=True)
    
        print("Experiment 2 results, full-query based candidates only:")
        if test_MPC_models:
            print("MPC: {}".format(mrr_mpc_exp2))

        if test_LM_models:
            print("MRR LM w/o features: {}".format(mrr_lm_without_features_exp2))
            print("MRR LM w/ features {}".format(mrr_lm_with_features_exp2))

        if calculate_prefix_count:
            if test_MPC_models:
                print("MRR MPC Divided: {}".format(mrr_mpc_divided_exp2))

            if test_LM_models:
                print("MRR LM w/o features Divided: {}".format(mrr_lm_divided_without_features_exp2))
                print("MRR LM w/ features Divided: {}".format(mrr_lm_divided_with_features_exp2))
    
    ## Create bargraph to see difference in distribution of suggested candidates for both experiments
    if suggested_candidates_plot:
        barWidth					= 0.25
        labels 						= (2,3,4,5,6,7,8,9,">=10")
        y_pos 						= np.arange(len(labels))
        y_pos2 						= [x + barWidth for x in y_pos]
        plt.figure(figsize=(8,4))
        plt.xticks(y_pos, labels)
        plt.xlabel('Number of candidates suggested')
        plt.ylabel('Distribution [%]')
        axes = plt.gca()
        axes.yaxis.grid()
        
        plt.bar(y_pos, ranks_mpc_ratio_exp1, align='center', width=barWidth, alpha=0.5, label='Experiment 1: Full-Query Based Candidates')
        plt.bar(y_pos2, ranks_mpc_ratio_exp2, align='center', width=barWidth, alpha=0.5, label='Experiment 2: Full-Query Based Candidates + Suffix Based Candidates')
        plt.legend()
        plt.title('Distribution of Suggested Candidate Suffixes')
        plt.savefig('CandidatesComparisonExp1-2.png')

main()



# Experiment 1:		Frequent	Rare	Unseen
# MPC					0.1529		0.4309	0
# LM w/o features		0.1747		0.2079	0.3278
# LM w/ featurs		0.4354		0.1947	0.5			
# 
# Experiment 2:		Frequent	Rare	Unseen
# MPC					0.1773		0.7782	0
# LM w/o features		0.1760		0.0748	0.1945
# LM w/ featurs		0.4133		0.2569	0.3304