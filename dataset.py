import numpy as np
import os
import pandas as pd
import string
import gc

class Dataset:
    def __init__(self):
        self.directory = "data"
        self.popular_suffixes_file = self.directory + "/popular_suffixes.npy"
        self.logs_directory = self.directory + "/logs"

    # Returns all popular suffixes in the query logs
    def get_popular_suffixes(self, overwrite=False):
        if overwrite or not os.path.isfile(self.popular_suffixes_file):
            suffixes = {}
            for filename in os.listdir(self.logs_directory):
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-03-01') & (df['QueryTime'] < '2006-05-01')]
                df["Query"] = df["Query"].str.replace('.com', ' com')
                df["Query"] = df["Query"].str.replace('[{}]'.format(string.punctuation), '')
                df["Query"] = df["Query"].str.lower()
                df = df.drop(df[df["Query"] == ""].index)
                for query in df["Query"].values:
                    words = str(query).split(" ")
                    for i in range(0, min(3, len(words))):
                        suffix = ' '.join(words[-(i+1):])
                        if suffix in suffixes:
                            suffixes[suffix] += 1
                        else:
                            suffixes[suffix] = 1
                print(len(suffixes))
                del df
                gc.collect()
            suffixes = list(suffixes.items())
            suffixes.sort(key=lambda x: x[1], reverse=True)
            suffixes = list(map(lambda x: x[0], suffixes))
            suffixes = suffixes[:100000]
            np.save(self.popular_suffixes_file, np.array(suffixes))
            return suffixes
        else:
            return np.load(self.popular_suffixes_file)
