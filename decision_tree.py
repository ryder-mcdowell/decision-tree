import pickle
import collections
import math
import operator

def entropy(data):
    category_frequencies = collections.Counter([category[-1] for category in data])

    def category_entropy(category_frequency):
        ratio = float(category_frequency) / len(data)
        return -1 * ratio * math.log(ratio, 2)

    return sum(category_entropy(category_frequency) for category_frequency in category_frequencies.values())


def best_feature_index_for_split(data):
    baseline = entropy(data)

    def feature_entropy(feature):
        def e(value):
            partitioned_data = [row for row in data if row[feature] == value]
            proportion = (float(len(partitioned_data)) / float(len(data)))
            return proportion * entropy(partitioned_data)
        return sum(e(value) for value in set([data_point[feature] for data_point in data]))

    features = len(data[0]) - 1
    information_gain = [baseline - feature_entropy(feature) for feature in range(features)]
    best_feature_index, best_gain = max(enumerate(information_gain), key=operator.itemgetter(1))
    return best_feature_index

if __name__ == '__main__':
    with open("data_rand", "rb") as f:
        L = pickle.load(f)
        print(best_feature_index_for_split(L))