import numpy as np
import csv
import random
import multiprocessing
from multiprocessing import Pool, freeze_support, set_start_method

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plotData(yTest, yPrediction, errVal):
    plt.figure(figsize = (6, 4), dpi=250)
    plt.plot([0, 0.6], [0, 0.6], ls="--", c=".1")
    plt.scatter(yTest, yPrediction, marker = 'o', s = 0.3, color='steelblue') # true values, predicted values, s is smaller so it is better visible

    plt.xlim(0, 0.6)
    plt.ylim(0, 0.6)

    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.02))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.02))

    plt.tick_params(axis='both', which='both', direction='in', right=False, top=False, labelsize=10, pad=6)

    plt.text(0.03, 0.57, f'MAE={errVal:.4f}', horizontalalignment='left', verticalalignment='top', fontsize=10)

    plt.gca().yaxis.get_major_ticks()[0].label1.set_visible(False) # hide 0
    plt.gca().xaxis.get_major_ticks()[0].label1.set_visible(False) # hide 0

    plt.xlabel(r'$Z_{\mathrm{spec}}$', fontsize=10)
    plt.ylabel(r'$Z_{\mathrm{phot}}$', fontsize=10)

    plt.savefig(f'scatterResult3.png', bbox_inches='tight', pad_inches=0.1)

def dataLabeling(): # sort data to arrays
    photometric = []
    specz = []

    csvData = f'data.csv'
    with open(csvData, 'r') as csvFile:
        reader = csv.reader(csvFile)
        next(reader)
        for i in reader:
            photometric.append([float(i[4]), float(i[5]), float(i[6]), float(i[7]), float(i[8]), float(i[9]), float(i[10]), float(i[11]), float(i[12])])
            specz.append(float(i[-1]))

    for i in range(len(photometric)):
        for j in range(len(photometric[i])):
            photometric[i][j] = np.log2(photometric[i][j] + 1)

    return photometric, specz


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, leaf_node_value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        self.leaf_node_value = leaf_node_value

class DecisionTree():
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, current_depth=0):
        x, y = dataset
        
        num_samples, num_features = np.shape(x)
        
        best_split = {}

        # smaples more than minimum and current depth is smaller equal to max depth
        if num_samples >= self.min_samples_split and current_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            
            if best_split and 'var_red' in best_split and best_split['var_red'] > 0:
                # print(np.shape(best_split['dataset_left'][0]), np.shape(best_split['dataset_right'][0]))
                
                # recursion
                left_subtree = self.build_tree((best_split["dataset_left"][0], best_split["dataset_left"][1]), current_depth + 1)
                right_subtree = self.build_tree((best_split["dataset_right"][0], best_split["dataset_right"][1]), current_depth + 1)
                
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["var_red"])
        # print(best_split["dataset_left"])
        leaf_value = self.calculate_leaf_value(y)
        return Node(leaf_node_value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_var_red = -float("inf")

        feature_values = dataset[0] # get the rows, previously was columns, but that is wrong --- PHOTOMETRIC DATA


        possible_thresholds = []
        for feature_index in range(num_features):
            feature_column = [x[feature_index] for x in feature_values]

            for val in range(len(feature_column)):
                if feature_column[val] not in possible_thresholds:
                    possible_thresholds.append(feature_column[val])
            # print(feature_column)
            possible_thresholds.sort()
            
            # print(possible_thresholds)

            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold) # params are dataset and singular photometric data value, return x, y
                
                if len(dataset_left[0]) > 0 and len(dataset_right[0]) > 0:
                    y, left_y, right_y = dataset[1], dataset_left[1], dataset_right[1] # REDSHIFT DATA
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # print(curr_var_red)
                    if curr_var_red > max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red

        return best_split


    def split(self, dataset, feature_index, threshold):        
        dataset_left_photo = []
        dataset_left_z = []
        dataset_right_photo = []
        dataset_right_z = []
        
        feature_values = dataset[0]
        target_values = dataset[1]

        for i in range(len(feature_values)):
            if len(feature_values) != len(target_values):
                break
            value_photo = np.array(feature_values[i])
            value_z = np.array(target_values[i])

            if value_photo[feature_index] <= threshold:
                dataset_left_photo.append(value_photo)
                dataset_left_z.append(value_z)
            else:
                dataset_right_photo.append(value_photo)
                dataset_right_z.append(value_z)
        # print(dataset_left_z)

        dataset_left = [dataset_left_photo, dataset_left_z]
        dataset_right = [dataset_right_photo, dataset_right_z]

        return dataset_left, dataset_right
    
    def variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    def calculate_leaf_value(self, y):
        val = np.mean(y)
        return val
    
    def print_tree(self, tree=None, indent=" "):
        if tree is None:
            tree = self.root
        if tree.leaf_node_value is not None:
            print('Leaf node: ', tree.leaf_node_value)
        else:
            print(tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, dataset):
        self.root = self.build_tree(dataset)
        print(self.root)

    
    def make_prediction(self, x, tree):
        
        if tree.leaf_node_value != None: 
            return tree.leaf_node_value
        feature_val = x[tree.feature_index]

        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def predict(self, x):
        ''' function to predict a single data point '''
        
        preditions = [self.make_prediction(val, self.root) for val in x]
        return preditions

class RandomForestRegressor:
    def __init__(self, n_trees=20, min_samples_split=2, max_depth=2):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []
    
    def bootstrap(self, dataset):
        bootstrap_indices = []
        x_train, y_train = dataset
        n_samples = len(x_train)

        for x in range(round(n_samples / 150)):
            bootstrap_indices.append(random.randint(0, n_samples - 1))
        
        # we need to return elements at specific indices
        x_bootstrap, y_bootstrap = [], []
        for x in bootstrap_indices:
            x_bootstrap.append(x_train[x])

        for y in bootstrap_indices:
            y_bootstrap.append(y_train[y])

        return x_bootstrap, y_bootstrap
    
    def fit_single_tree(self, dataset):
        x_boots, y_boots = dataset
        tree = DecisionTree(min_samples_split=3, max_depth=15)
        tree.fit((x_boots, y_boots))
        return tree
    
    def fit(self, dataset):

        self.trees = []

        # with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        #     futures = [executor.submit(self.fit_single_tree, self.bootstrap(dataset)) for _ in range(self.n_trees)]
        #     self.trees = [future.result() for future in futures]
        
        x_bootstrap_datasets = [self.bootstrap(dataset) for _ in range(self.n_trees)]

        with Pool(processes=18) as pool:
            self.trees = pool.map(self.fit_single_tree, x_bootstrap_datasets)
            # children = multiprocessing.active_children()
            # print(children)
    
    def predict(self, x):
        output_from_trees = np.zeros((len(x), len(self.trees)))

        for i, tree in enumerate(self.trees):
            output_from_trees[:, i] = tree.predict(x)
        return np.mean(output_from_trees, axis=1)

if __name__ == "__main__":
    x, y = dataLabeling()
    # x = x[5000:10000]
    # y = y[5000:10000]

    x = x[:194960]
    y = y[:194960]

    x_test, y_test = dataLabeling()
    x_test, y_test = x_test[194960:], y_test[194960:]

    forest = RandomForestRegressor(n_trees=15)
    forest.fit((x, y))
    pred = forest.predict(x_test)
    # print(pred)

    from sklearn.metrics import mean_absolute_error

    # plotData(y_test, pred, mean_absolute_error(y_test, pred))

# File "C:\Program Files\Python312\Lib\multiprocessing\spawn.py", line 164, in get_preparation_data
#     _check_not_importing_main()
#   File "C:\Program Files\Python312\Lib\multiprocessing\spawn.py", line 140, in _check_not_importing_main
#     raise RuntimeError('''
# RuntimeError:
#         An attempt has been made to start a new process before the
#         current process has finished its bootstrapping phase.

#         This probably means that you are not using fork to start your
#         child processes and you have forgotten to use the proper idiom
#         in the main module:

#             if __name__ == '__main__':
#                 freeze_support()
#                 ...

#         The "freeze_support()" line can be omitted if the program
#         is not going to be frozen to produce an executable.

#         To fix this issue, refer to the "Safe importing of main module"
#         section in https://docs.python.org/3/library/multiprocessing.html
