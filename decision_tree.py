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


def potential_leaf_node(data):
    counts = collections.Counter([i[-1] for i in data])
    return counts.most_common(1)[0]


def create_tree(data, label):
    category, count = potential_leaf_node(data)

    if count == len(data):
        return category

    node = {}
    feature = best_feature_index_for_split(data)
    feature_label = label[feature]
    node[feature_label] = {}
    classes = set([data_point[feature] for data_point in data])
    
    for c in classes:
        partitioned_data = [data_point for data_point in data if data_point[feature] == c]
        node[feature_label][c] = create_tree(partitioned_data, label)

    return node


def classify(tree, label, data):
    root = list(tree.keys())[0]
    node = tree[root]
    index = label.index(root)
    for key in node.keys():
        if data[index] == key:
            if isinstance(node[key], dict):
                return classify(node[key], label, data)
            else:
                return node[key]


def as_rule_str(tree, label, ident=0):
    space_ident = ' ' * ident
    s = space_ident
    root = list(tree.keys())[0]
    node = tree[root]
    index = label.index(root)
    
    for key in node.keys():
        s += 'if ' + label[index] + ' = ' + str(key)
        if isinstance(node[key], dict):
            s += ':\n' + space_ident + as_rule_str(node[key], label, ident + 1)
        else:
            s += ' then ' + str(node[key]) + ('.\n' if ident == 0 else ', ')

    if s[-2:] == ', ':
        s = s[:-2]

    s += '\n'

    return s


if __name__ == '__main__':
    with open("data_rand", "rb") as f:
        L = pickle.load(f)
        #data = [[0, 0, False], [1, 0, False], [0, 1, True], [1, 1, True]]
        label = ['x', 'y', 'out']
        tree = create_tree(L, label)
        print(as_rule_str(tree, label))
        print(classify(tree, label, [1, 1]))