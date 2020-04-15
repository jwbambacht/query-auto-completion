import numpy as np
import os
import pandas as pd
import string
import gc
import random
import struct

class Data:
    def __init__(self):
        random.seed(42)
        self.directory = "data"
        self.popular_suffixes_file = self.directory + "/popular_suffixes.txt"
        self.popular_prefixes_file = self.directory + "/popular_prefixes.txt"
        self.popular_queries_file = self.directory + "/popular_queries.txt"
        self.candidate_queries_file = self.directory + "/candidate_queries_{}.txt"
        self.unique_queries_file = self.directory + "/unique_queries_{}.txt"
        self.candidate_frequencies_file = self.directory + "/candidate_frequencies_{}.txt"
        self.training_samples_file = self.directory + "/training_samples_{}.txt"
        self.testing_samples_file = self.directory + "/testing_samples_{}.txt"
        self.n_gram_frequency_file = self.directory + "/{}_gram_frequency.txt"
        self.logs_directory = self.directory + "/logs"

    # Method to normalize the queries
    @staticmethod
    def normalize_queries(df):
        df["Query"] = df["Query"].map(str)
        df["Query"] = df["Query"].str.replace('www.', 'www ')
        df["Query"] = df["Query"].str.replace('.com', ' com')
        df["Query"] = df["Query"].str.replace('[{}]'.format(string.punctuation), '')
        df["Query"] = df["Query"].str.lower()
        mask = df["Query"] != ""
        df = df[mask]
        return df

    # Method to get the popular suffixes
    def get_popular_suffixes(self, overwrite=False):
        if overwrite or not os.path.isfile(self.popular_suffixes_file):
            print("Generating most popular suffixes")
            suffixes = {}
            for filename in os.listdir(self.logs_directory):
                print(filename)
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-03-01') & (df['QueryTime'] < '2006-05-01')]
                df = self.normalize_queries(df)
                for query in df["Query"].values:
                    words = str(query).split(" ")
                    while '' in words:
                        words.remove('')
                    for i in range(0, len(words)):
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
            suffixes = suffixes[:100000]
            f = open(self.popular_suffixes_file, "w")
            for suffix in suffixes:
                f.write(suffix)
                f.write("\n")
            f.close()
            return suffixes
        else:
            suffixes = []
            f = open(self.popular_suffixes_file, "r", encoding='latin-1')
            for line in f.readlines():
                suffixes += [line[:-1]]
            f.close()
            return suffixes

    # Method to get the popular queries from the files
    def get_popular_queries(self, overwrite=False):
        if overwrite or not os.path.isfile(self.popular_queries_file):
            queries = {}
            for index, filename in enumerate(os.listdir(self.logs_directory)):
                print(filename)
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-03-01') & (df['QueryTime'] < '2006-05-01')]
                df = self.normalize_queries(df)
                for query in df["Query"].values:
                    if query in queries:
                        queries[query] += 1
                    else:
                        queries[query] = 1
                del df
                gc.collect()
            queries = list(queries.items())
            queries.sort(key=lambda x: x[1], reverse=True)
            queries = list(map(lambda x: x[0], queries))
            queries = queries[:724340]
            f = open(self.popular_queries_file, "w")
            for query in queries:
                f.write(query)
                f.write("\n")
            f.close()
            return queries
        else:
            queries = []
            f = open(self.popular_queries_file, "r", encoding='latin-1')
            for line in f.readlines():
                queries += [line[:-1]]
            f.close()
            return queries

    # Return all candidate queries
    def get_candidate_queries(self, n, overwrite=False):
        if overwrite or not os.path.isfile(self.candidate_queries_file.format(n)):
            print("Generating all candidate queries")
            suffixes = self.get_popular_suffixes()[:n]
            queries = self.get_popular_queries()
            candidates = []
            for index, query in enumerate(queries):
                # Each 1000 iterations, make sure the items in the list are unique
                if index % 1000 == 0:
                    print(index)
                    candidates = list(set(candidates))

                candidates += [query]

                query_split = query.split(" ")
                if query_split[-1] == "":
                    end_term = ' '.join(query_split[-2:])
                else:
                    end_term = query_split[-1]
                # Find all suffixes that fit the end term of the query
                for suffix in suffixes:
                    if suffix.startswith(end_term):
                        candidate = query + suffix[len(end_term):]
                        candidates += [candidate]
            candidates = list(set(candidates))
            print("Created {} candidates".format(len(candidates)))
            f = open(self.candidate_queries_file.format(n), "w")
            for index, candidate in enumerate(candidates):
                f.write(candidate)
                f.write("\n")
            f.close()
            return candidates
        else:
            candidates = []
            f = open(self.candidate_queries_file.format(n), "r", encoding='latin-1')
            for line in f.readlines():
                candidates += [line[:-1]]
            f.close()
            return candidates

    # Get frequencies of the candidates
    def get_candidate_frequencies(self, n, overwrite=False):
        if overwrite or not os.path.isfile(self.candidate_frequencies_file.format(n)):
            frequencies = {c: 0 for c in self.get_candidate_queries(n)}
            for filename in os.listdir(self.logs_directory):
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-03-01') & (df['QueryTime'] < '2006-05-01')]
                df = self.normalize_queries(df)
                for query in df["Query"].values:
                    query_split = query.split(" ")
                    if query_split[-1] == "":
                        query = ' '.join(query_split[:-1])
                    if query in frequencies:
                        frequencies[query] += 1
                del df
                gc.collect()
            f = open(self.candidate_frequencies_file.format(n), "w")
            for candidate, freq in frequencies.items():
                f.write("{},{}\n".format(candidate, str(freq)))
            f.close()
            return frequencies
        else:
            frequencies = {}
            f = open(self.candidate_frequencies_file.format(n), "r", encoding='latin-1')
            for line in f.readlines():
                split = line.split(",")
                frequencies[split[0]] = int(split[1])
            f.close()
            return frequencies

    # Get n-gram frequencies
    def n_gram_frequency(self, n, overwrite=False):
        if overwrite or not os.path.isfile(self.n_gram_frequency_file.format(n)):
            print("Generating the {}-gram frequency file".format(n))
            frequency = {}
            for filename in os.listdir(self.logs_directory):
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-03-01') & (df['QueryTime'] < '2006-05-01')]
                df = self.normalize_queries(df)
                for query in df["Query"].values:
                    query_split = query.split(" ")
                    while '' in query_split:
                        query_split.remove('')
                    n_grams = [' '.join(query_split[i:i+n]) for i in range(len(query_split) - (n - 1))]
                    for n_gram in n_grams:
                        if n_gram in frequency:
                            frequency[n_gram] += 1
                        else:
                            frequency[n_gram] = 1
                del df
                gc.collect()
                f = open(self.n_gram_frequency_file.format(n), "w")
                for n_gram, freq in frequency.items():
                    f.write("{},{}\n".format(n_gram, freq))
                f.close()
                return frequency
        else:
            frequency = {}
            f = open(self.n_gram_frequency_file.format(n), "r", encoding='latin-1')
            for line in f.readlines():
                split = line.split(",")
                frequency[split[0]] = int(split[1])
            f.close()
            return frequency

    @staticmethod
    def get_possible_candidates(prefix, candidates, query):
        index = 0
        # Find starting index
        while prefix >= candidates[index] or candidates[index].startswith(prefix):
            index += 10000
            if index > len(candidates) - 1:
                index = len(candidates) - 1
                break
        end_index = index

        index = max(0, index - 10000)

        while prefix <= candidates[index]:
            index -= 10000
            if index < 0:
                index = 0
                break

        return [q for q in candidates[index:end_index] if q.startswith(prefix) and q != query]

    # Get the training samples
    def get_training_samples(self, n, overwrite=False):
        if overwrite or not os.path.isfile(self.training_samples_file.format(n)):
            print("Generating the training samples")
            candidate_frequencies = self.get_candidate_frequencies(n)
            candidates = sorted(candidate_frequencies.keys())
            n_gram_frequencies = [self.n_gram_frequency(i) for i in range(1, 7)]
            queries = []
            prefixes = []
            prefix_ids = []
            all_features = []
            y = []
            for filename in os.listdir(self.logs_directory):
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-05-01') & (df['QueryTime'] < '2006-05-15')]
                df = self.normalize_queries(df)
                df = df[df["Query"].isin(candidates)]
                queries += list(df["Query"])
                queries = list(set(queries))
                del df
                gc.collect()
            print("Found {} queries for training".format(len(queries)))
            for index, query in enumerate(queries):
                if index % 1000 == 0:
                    print(index)
                # If the query is in the candidates
                query_split = query.split(" ")
                if len(query_split) > 1:
                    last_words = ' ' + ' '.join(query_split[1:])
                    prefix = query_split[0] + last_words[:random.randint(0, len(last_words) - 1)]
                    possible_candidates = Data.get_possible_candidates(prefix, candidates, query)
                    if len(possible_candidates) > 0:
                        if prefix in prefixes:
                            prefix_id = prefixes.index(prefix)
                        else:
                            prefixes += [prefix]
                            prefix_id = len(prefixes) - 1
                        prefix_ids += [prefix_id]
                        all_features += [
                            self.get_features(prefix, query, candidate_frequencies[query], n_gram_frequencies)]
                        y += [1]
                        if len(possible_candidates) > 9:
                            possible_candidates = list(
                                map(lambda x: (x, candidate_frequencies[x]), possible_candidates))
                            possible_candidates.sort(key=lambda x: x[1], reverse=True)
                            possible_candidates = possible_candidates[:9]
                        else:
                            possible_candidates = list(
                                map(lambda x: (x, candidate_frequencies[x]), possible_candidates))
                        for full_query, frequency in possible_candidates:
                            prefix_ids += [prefix_id]
                            all_features += [self.get_features(prefix, full_query, frequency, n_gram_frequencies)]
                            y += [0]
            all_features = list(map(lambda x: ",".join(list(map(str, x))), all_features))
            f = open(self.training_samples_file.format(n), "w")
            for i in range(len(prefix_ids)):
                f.write("{},{},{}".format(prefix_ids[i], all_features[i], y[i]))
                f.write("\n")
            f.close()
            return self.get_training_samples(n, overwrite=False)
        else:
            training_samples = []
            f = open(self.training_samples_file.format(n), encoding='latin-1')
            for line in f.readlines():
                features = list(map(int, line.split(",")))
                if len(features) > 0:
                    training_samples += [features]
            return training_samples

    def get_testing_samples(self, n, overwrite=False):
        if overwrite or not os.path.isfile(self.testing_samples_file.format(n)):
            print("Generating testing samples")
            candidate_frequencies = self.get_candidate_frequencies(n)
            candidates = sorted(candidate_frequencies.keys())

            n_gram_frequencies = [self.n_gram_frequency(i) for i in range(1, 7)]
            queries = []
            for filename in os.listdir(self.logs_directory):
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-05-15') & (df['QueryTime'] < '2006-05-29')]
                df = self.normalize_queries(df)
                df = df[df["Query"].isin(candidates)]
                queries += list(df["Query"])
                queries = list(set(queries))
                del df
                gc.collect()
            print("Found {} queries for testing".format(len(queries)))

            prefixes = []
            full_queries = []
            prefix_ids = []
            all_features = []
            classes = []
            y = []

            for index, query in enumerate(queries):
                if index % 1000 == 0:
                    print(index)
                query_split = query.split(" ")
                if len(query_split) > 1:
                    last_words = ' ' + ' '.join(query_split[1:])
                    prefix = query_split[0] + last_words[:random.randint(0, len(last_words) - 1)]
                    possible_candidates = Data.get_possible_candidates(prefix, candidates, query)
                    if len(possible_candidates) > 0:
                        if prefix in prefixes:
                            prefix_id = prefixes.index(prefix)
                        else:
                            prefixes += [prefix]
                            prefix_id = len(prefixes) - 1
                        if len(possible_candidates) > 9:
                            possible_candidates = list(
                                map(lambda x: (x, candidate_frequencies[x]), possible_candidates))
                            possible_candidates.sort(key=lambda x: x[1], reverse=True)
                            possible_candidates = possible_candidates[:9]
                        else:
                            possible_candidates = list(
                                map(lambda x: (x, candidate_frequencies[x]), possible_candidates))
                        prefix_ids += [prefix_id]
                        all_features += [
                            self.get_features(prefix, query, candidate_frequencies[query], n_gram_frequencies)]
                        classes += [prefix]
                        full_queries += [query]
                        y += [1]
                        for candidate, frequency in possible_candidates:
                            prefix_ids += [prefix_id]
                            all_features += [self.get_features(prefix, candidate, frequency, n_gram_frequencies)]
                            classes += [" "]
                            full_queries += [candidate]
                            y += [0]
            all_features = list(map(lambda x: ','.join(list(map(str, x))), all_features))
            f = open(self.testing_samples_file.format(n), "w")
            for i in range(len(prefix_ids)):
                f.write("{},{},{},{},{}\n".format(classes[i], full_queries[i], prefix_ids[i], all_features[i], y[i]))
            f.close()
            return self.get_testing_samples(n)
        else:
            testing_samples = []
            f = open(self.testing_samples_file.format(n), encoding='latin-1')
            for line in f.readlines():
                features = list(map(int, line.split(",")[2:]))
                if len(features) > 0:
                    testing_samples += [features]
            return testing_samples

    @staticmethod
    def get_features(prefix, full_query, candidate_frequency, n_gram_frequencies):
        features = []
        prefix_split = prefix.split(" ")
        if prefix_split[-1] == "":
            end_term = prefix_split[-2:]
        else:
            end_term = prefix_split[-1]
        end_term_start = len(prefix) - len(end_term)
        suffix = full_query[end_term_start:]

        # N-gram based features
        for n in range(1, 7):
            query_split = full_query.split(" ")
            if query_split[-1] == "":
                query_split = query_split[:-1]
            n_grams = [' '.join(query_split[i:i+n]) for i in range(len(query_split) - (n-1))]
            n_gram_sum = 0
            for n_gram in n_grams:
                if n_gram in n_gram_frequencies[n - 1]:
                    n_gram_sum += n_gram_frequencies[n - 1][n_gram]
            features += [n_gram_sum]

        # Candidate frequency feature
        features += [candidate_frequency]

        # Character length features
        features += [len(prefix), len(suffix), len(full_query)]

        # Word length features
        if prefix_split[-1] == "":
            features += [len(prefix_split) - 1]
        else:
            features += [len(prefix_split)]

        suffix_split = suffix.split(" ")
        if suffix_split[-1] == "":
            features += [len(suffix_split) - 1]
        else:
            features += [len(suffix_split)]

        full_query_split = full_query.split(" ")
        if full_query_split[-1] == "":
            features += [len(full_query_split) - 1]
        else:
            features += [len(full_query_split)]

        # Ends with space feature
        if prefix_split[-1] == "":
            features += [1]
        else:
            features += [0]

        return features