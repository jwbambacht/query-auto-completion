import numpy as np
import os
import pandas as pd
import string
import gc
import random

class Dataset:
    def __init__(self):
        random.seed(42)
        self.directory = "data"
        self.popular_suffixes_file = self.directory + "/popular_suffixes.npy"
        self.popular_prefixes_file = self.directory + "/popular_prefixes.npy"
        self.popular_queries_file = self.directory + "/popular_queries.npy"
        self.candidate_queries_file = self.directory + "/candidate_queries.npy"
        self.unique_queries_file = self.directory + "/unique_queries_{}.npy"
        self.candidate_frequencies_file = self.directory + "/candidate_frequencies.npy"
        self.training_samples_file = self.directory + "/training_samples.npy"
        self.testing_samples_file = self.directory + "/testing_samples.npy"
        self.n_gram_frequency_file = self.directory + "/{}_gram_frequency.npy"
        self.logs_directory = self.directory + "/logs"

    # Returns all popular suffixes in the `background` of the query logs
    def get_popular_suffixes(self, overwrite=False):
        if overwrite or not os.path.isfile(self.popular_suffixes_file):
            suffixes = {}
            for filename in os.listdir(self.logs_directory):
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-03-01') & (df['QueryTime'] < '2006-05-01')]
                df = self.normalize_queries(df)
                for query in df["Query"].values:
                    words = str(query).split(" ")
                    if words[-1] == "":
                        words = words[:-1]
                    for i in range(0, min(3, len(words))):
                        suffix = ' '.join(words[-(i+1):])
                        if suffix in suffixes:
                            suffixes[suffix] += 1
                        else:
                            suffixes[suffix] = 1
                del df
                gc.collect()
            suffixes = list(suffixes.items())
            suffixes.sort(key=lambda x: x[1], reverse=True)
            suffixes = list(map(lambda x: x[0], suffixes))
            suffixes = np.array(suffixes[:10000])
            np.save(self.popular_suffixes_file, suffixes)
            return suffixes
        else:
            return np.load(self.popular_suffixes_file)

    def get_popular_prefixes(self, overwrite=False):
        if overwrite or not os.path.isfile(self.popular_prefixes_file):
            prefixes = {}
            for index, filename in enumerate(os.listdir(self.logs_directory)):
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-03-01') & (df['QueryTime'] < '2006-05-01')]
                df = self.normalize_queries(df)
                for query in df["Query"].values:
                    words = str(query).split(" ")
                    if words[-1] == "":
                        words = words[:-1]
                    for i in range(len(words)):
                        prefix = ' '.join(words[:i+1])
                        if prefix in prefixes:
                            prefixes[prefix] += 1
                        else:
                            prefixes[prefix] = 1
                if len(prefixes) > 3000000:
                    prefixes = {k: v for k, v in prefixes.items() if v not in {1, 2}}
                del df
                gc.collect()
            prefixes = list(prefixes.items())
            print(len(prefixes))
            prefixes.sort(key=lambda x: x[1], reverse=True)
            prefixes = prefixes[:100000]
            print(prefixes[-1])
            f = open(self.popular_prefixes_file, "w")
            for index, prefix in enumerate(prefixes):
                f.write("{},{}".format(prefix[0],prefix[1]))
                if index != len(prefixes) - 1:
                    f.write("\n")
            f.close()
            return prefixes
        else:
            prefixes = []
            f = open(self.popular_prefixes_file, "r")
            for line in f.readlines():
                prefixes += [line.split(",")[0]]
            f.close()
            return prefixes
