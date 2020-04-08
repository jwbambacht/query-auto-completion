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

    # Returns all popular queries in the `background` of the query logs
    def get_popular_queries(self, overwrite=False):
        if overwrite or not os.path.isfile(self.popular_queries_file):
            queries = {}
            for index, filename in enumerate(os.listdir(self.logs_directory)):
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-03-01') & (df['QueryTime'] < '2006-05-01')]
                df = self.normalize_queries(df)
                for query in df["Query"].values:
                    query = str(query)
                    if query in queries:
                        queries[query] += 1
                    else:
                        queries[query] = 1
                if index == 5:
                    queries = {k: v for k, v in queries.items() if v not in {1, 2}}
                del df
                gc.collect()
            queries = list(queries.items())
            queries.sort(key=lambda x: x[1], reverse=True)
            queries = list(map(lambda x: x[0], queries))
            queries = np.array(queries[:100000])
            np.save(self.popular_queries_file, queries)
            return queries
        else:
            return np.load(self.popular_queries_file)

    # Return all candidate queries
    def get_candidate_queries(self, overwrite=False):
        if overwrite or not os.path.isfile(self.candidate_queries_file):
            print("Generating all candidate queries")
            suffixes = self.get_popular_suffixes()[:3500]
            queries = self.get_popular_queries()
            candidates = []

            for index, query in enumerate(queries):
                candidates += [query]

                # Each 1000 iterations, make sure the items in the list are unique
                if index % 1000 == 0:
                    candidates = list(set(candidates))

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
            del suffixes
            del queries
            gc.collect()
            candidates = list(set(candidates))
            print("Created {} candidates".format(len(candidates)))
            f = open(self.candidate_queries_file, "w")
            for index, candidate in enumerate(candidates):
                f.write(candidate)
                f.write("\n")
            f.close()
            return candidates
        else:
            candidates = []
            f = open(self.candidate_queries_file, "r")
            for line in f.readlines():
                candidates += [line[:-1]]
            f.close()
            return candidates

    def get_unique_queries(self, part=1, overwrite=False):
        if overwrite or not os.path.isfile(self.unique_queries_file.format(part)):
            for index, filename in enumerate(os.listdir(self.logs_directory)):
                queries = []
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-03-01') & (df['QueryTime'] < '2006-05-01')]
                df = self.normalize_queries(df)
                unique_queries = df["Query"].unique().tolist()
                queries += unique_queries
                queries = list(set(queries))
                print(len(queries))
                del df
                gc.collect()
                f = open(self.unique_queries_file.format(index+1), "w")
                for index, query in enumerate(queries):
                    f.write(query)
                    if index != len(queries) - 1:
                        f.write("\n")
                f.close()
            return self.get_unique_queries()
        else:
            queries = []
            f = open(self.unique_queries_file.format(part), "r")
            for line in f.readlines():
                queries += [line[:-1]]
            f.close()
            return queries

    def get_candidate_frequencies(self, candidates, overwrite=False):
        if overwrite or not os.path.isfile(self.candidate_frequencies_file):
            frequencies = {c: 0 for c in candidates}
            for filename in os.listdir(self.logs_directory):
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-03-01') & (df['QueryTime'] < '2006-05-01')]
                df = self.normalize_queries(df)
                for query in df["Query"].values:
                    if query in frequencies:
                        frequencies[query] += 1
                del df
                gc.collect()

            frequencies = list(map(lambda x: frequencies[x], candidates))
            f = open(self.candidate_frequencies_file, "w")
            for index, freq in enumerate(frequencies):
                f.write(str(freq))
                if index != len(frequencies) - 1:
                    f.write("\n")
            f.close()
            return frequencies
        else:
            frequencies = []
            f = open(self.candidate_frequencies_file, "r")
            for line in f.readlines():
                frequencies += [int(line)]
            f.close()
            return frequencies

    # Get the training samples
    def get_training_samples(self, overwrite=False):
        if overwrite or not os.path.isfile(self.training_samples_file):
            print("Generating the training samples")
            open(self.training_samples_file, "w").close()
            candidates = self.get_candidate_queries()
            candidate_frequencies = self.get_candidate_frequencies(candidates)
            candidate_frequencies = {candidates[i]: candidate_frequencies[i] for i in range(len(candidates))}
            n_gram_frequencies = [self.n_gram_frequency(i) for i in range(1, 7)]
            queries = []
            prefixes = []
            prefix_ids = []
            all_features = []
            y = []
            for i in range(6):
                n_gram_frequencies[i] = {item[0]: item[1] for item in n_gram_frequencies[i]}
            for filename in os.listdir(self.logs_directory):
                df = pd.read_csv("{}/{}".format(self.logs_directory, filename), sep="\t")
                df = df[(df['QueryTime'] > '2006-05-01') & (df['QueryTime'] < '2006-05-15')]
                df = self.normalize_queries(df)
                df = df[df["Query"].isin(candidates)]
                queries += list(df["Query"])
                queries = list(set(queries))
                del df
                gc.collect()
            count = 0
            print(len(queries))
            for query in queries:
                if count % 1000 == 0:
                    print(count)
                count += 1
                # If the query is in the candidates
                query_split = query.split(" ")
                if len(query_split) > 1:
                    last_words = ' ' + ' '.join(query_split[1:])
                    prefix = query_split[0] + last_words[:random.randint(0, len(last_words))]
                    possible_candidates = [q for q in candidates if q.startswith(prefix) and q != query]
                    if len(possible_candidates) > 0:
                        if prefix in prefixes:
                            prefix_id = prefixes.index(prefix)
                        else:
                            prefixes += [prefix]
                            prefix_id = len(prefixes) - 1
                        prefix_ids += [prefix_id]
                        all_features += [self.get_features(prefix, query, candidate_frequencies[query], n_gram_frequencies)]
                        y += [1]
                        if len(possible_candidates) > 9:
                            possible_candidates = list(
                                map(lambda x: (x, candidate_frequencies[x]), possible_candidates))
                            possible_candidates.sort(key=lambda x: x[1], reverse=True)
                            possible_candidates = possible_candidates[:9]
                        else:
                            possible_candidates = list(map(lambda x: (x, candidate_frequencies[x]), possible_candidates))
                        for full_query, frequency in possible_candidates:
                            prefix_ids += [prefix_id]
                            all_features += [self.get_features(prefix, full_query, frequency, n_gram_frequencies)]
                            y += [0]
                if len(prefix_ids) >= 100:
                    self.write_to_file(self.training_samples_file, prefix_ids, all_features, y)
                    prefix_ids = []
                    all_features = []
                    y = []
            self.write_to_file(self.training_samples_file, prefix_ids, all_features, y)
            return self.get_training_samples(overwrite=False)
        else:
            training_samples = []
            f = open(self.training_samples_file)
            for line in f.readlines():
                features = list(map(int, line.split(",")))
                if len(features) > 0:
                    training_samples += [features]
            return training_samples

    def get_testing_samples(self, overwrite=False):
        if overwrite or not os.path.isfile(self.testing_samples_file):
            print("Generating testing samples")
            candidates = self.get_candidate_queries()
            candidate_frequencies = self.get_candidate_frequencies(candidates)
            candidate_frequencies = {candidates[i]: candidate_frequencies[i] for i in range(len(candidates))}
            n_gram_frequencies = [self.n_gram_frequency(i) for i in range(1, 7)]
            for i in range(6):
                n_gram_frequencies[i] = {item[0]: item[1] for item in n_gram_frequencies[i]}
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
            print(len(queries))

            prefixes = []
            full_queries = []
            prefix_ids = []
            all_features = []
            classes = []
            y = []
            for index, query in enumerate(queries):
                if index % 100 == 0:
                    print(index)
                query_split = query.split(" ")
                if len(query_split) > 1:
                    last_words = ' ' + ' '.join(query_split[1:])
                    prefix = query_split[0] + last_words[:random.randint(0, len(last_words)-1)]
                    print(prefix)
                    possible_candidates = [q for q in candidates if q.startswith(prefix) and q != query]
                    print(len(possible_candidates))
                    if len(possible_candidates) > 0:
                        if prefix in prefixes:
                            prefix_id = prefixes.index(prefix)
                        else:
                            prefixes += [prefix]
                            prefix_id = len(prefixes) - 1
                        if len(possible_candidates) > 9:
                            possible_candidates = list(map(lambda x: (x, candidate_frequencies[x]), possible_candidates))
                            possible_candidates.sort(key=lambda x: x[1], reverse=True)
                            possible_candidates = possible_candidates[:9]
                        else:
                            possible_candidates = list(map(lambda x: (x, candidate_frequencies[x]), possible_candidates))
                        prefix_ids += [prefix_id]
                        all_features += [self.get_features(prefix, query, candidate_frequencies[query], n_gram_frequencies)]
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
            f = open(self.testing_samples_file, "w")
            for i in range(len(prefix_ids)):
                f.write("{},{},{},{},{}".format(classes[i], full_queries[i], prefix_ids[i], all_features[i], y[i]))
                f.write("\n")
            f.close()
            return self.get_testing_samples()
        else:
            testing_samples = []
            f = open(self.testing_samples_file)
            for line in f.readlines():
                split_line = line.split(",")
                features = list(map(int, line.split(",")[2:]))
                if len(features) > 0:
                    testing_samples += [features]
            return testing_samples

    @staticmethod
    def write_to_file(filename, prefix_ids, features, y):
        features = list(map(lambda x: ",".join(list(map(str, x))), features))
        f = open(filename, "a")
        for i in range(len(prefix_ids)):
            f.write("{},{},{}".format(prefix_ids[i], features[i], y[i]))
            f.write("\n")
        f.flush()
        f.close()


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
                    if query_split[-1] == "":
                        query_split = query_split[:-1]
                    n_grams = [' '.join(query_split[i:i+n]) for i in range(len(query_split) - (n - 1))]
                    for n_gram in n_grams:
                        if n_gram in frequency:
                            frequency[n_gram] += 1
                        else:
                            frequency[n_gram] = 1
                del df
                gc.collect()
                frequency = list(frequency.items())
                f = open(self.n_gram_frequency_file.format(n), "w")
                for index, freq in enumerate(frequency):
                    f.write("{},{}".format(freq[0], freq[1]))
                    if index != len(frequency) - 1:
                        f.write("\n")
                f.close()
                return frequency
        else:
            frequency = []
            f = open(self.n_gram_frequency_file.format(n), "r")
            for line in f.readlines():
                freq = line.split(",")
                frequency += [(freq[0], int(freq[1]))]
            f.close()
            return frequency

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
            n_grams = [' '.join(query_split[i:i+n]) for i in range(len(query_split) - (n - 1))]
            n_gram_sum = 0
            for n_gram in n_grams:
                if n_gram in n_gram_frequencies[n-1]:
                    n_gram_sum += n_gram_frequencies[n-1][n_gram]
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
