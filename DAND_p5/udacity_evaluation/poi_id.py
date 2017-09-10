#!/usr/bin/python

import warnings
warnings.filterwarnings("ignore") # hide warnings in cell output

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# all features (except for the email address) will be used for the first 
# analysis steps, the final feature list will be specified at the end before it 
# is dumped
features_list = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 
            'director_fees', 'exercised_stock_options', 'expenses', 
            'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', 
            'restricted_stock_deferred', 'salary', 'total_payments', 
            'total_stock_value', 'from_messages', 'from_poi_to_this_person', 
            'from_this_person_to_poi', 'shared_receipt_with_poi', 'to_messages',
            'restricted_stock_fraction', 'poi_email_fraction']

### Load the dictionary containing the dataset
with open("enron_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
for k in ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']:
	data_dict.pop(k, None)

### Task 3: Create new feature(s)
for k, v in data_dict.items():
	# ratio of stock that is restricted to total stock value
	restricted_stock = data_dict[k]['restricted_stock']
	total_stock = data_dict[k]['total_stock_value']
	
	# restricted_stock and total_stock avialable
	if  isinstance(restricted_stock, int) and isinstance(total_stock, int):
		data_dict[k]['restricted_stock_fraction'] = restricted_stock / total_stock
	# stock data not available
	else:
		data_dict[k]['restricted_stock_fraction'] = 'NaN'

	# ratio of poi emails to total emails
	total_poi = (data_dict[k]['from_poi_to_this_person'] + 
	             data_dict[k]['from_this_person_to_poi'])
	total_email = (data_dict[k]['from_messages'] +
	               data_dict[k]['to_messages'])
	
	# total_poi and total_email available
	if isinstance(total_poi, int) and isinstance(total_email, int):
		data_dict[k]['poi_email_fraction'] = total_poi / total_email
	# total_poi not defined or no email data available
	else:
		data_dict[k]['poi_email_fraction'] = 'NaN'

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# convert label list to numpy array
labels = np.array(labels)

# feature scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### initial model screening


def get_classifier_scores(clf, X, y, random_state=None):
    """Train classifier and evaluate its accuracy, precision and recall score. 
    Uses 100 stratified and shuffled splits with a split ratio of 33% for 
    crross-validation and calculates the mean values for each metric.

    Args:
        clf: Scikit-learn classifier object.
        X: Feature numpy array or DataFrame.
        y: Label numpy array Dataframe.
        random_state: Seed for randomizer in the StratifiedShuffleSplit()
            function.

    Returns:
        List containing the mean accuracy, precision and recall for the 
        evaluated classifier.

        [accuracy, precision, recall]
    """
    
    # check if data set is in a dataframe, if so convert it to a numpy array
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    accuracies, precisions, recalls = [], [], []
    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.33, random_state=42)
    
    for train_ixs, test_ixs in sss.split(X, y):
        X_train, X_test = X[train_ixs, :], X[test_ixs, :]
        y_train, y_test = y[train_ixs] , y[test_ixs]
    
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        accuracies.append(clf.score(X_test, y_test))
        precisions.append(precision_score(y_test, pred))
        recalls.append(recall_score(y_test, pred))
    
    return [np.mean(accuracies), np.mean(precisions), np.mean(recalls)]


def get_multi_classifier_scores(names, classifiers, X, y, random_state=None):
    """Calculate the accuracy, precision and recall scores for multiple 
    classifiers. Uses 100 stratified and shuffled splits with a split ratio of 
    33% for crross-validation and calculates the mean values for each metric.

    Args:
        names: List of classifier names.
        classifiers: List of scikit-learn classifier objects.
        X: Feature numpy array or DataFrame.
        y: Label numpy array Dataframe.
        random_state: Seed for randomizer in the StratifiedShuffleSplit()
            function.

    Returns:
        Dictionary containing dictionairies for each classifier name key with
        accuracy, precision and recall key-value pairs.

        {clf_name1: {accuracy: ..., precision: ..., recall: ...},
         clf_name2: {...}, 
         ...}
    """
    
    clf_scores = OrderedDict()
    for n, clf in zip(names, classifiers):
        clf_scores[n] = OrderedDict()
        scores = get_classifier_scores(clf, X, y, random_state=random_state)
        
        clf_scores[n]['accuracy'] = scores[0]
        clf_scores[n]['precision'] = scores[1]
        clf_scores[n]['recall'] = scores[2]
    
    return clf_scores


def print_classifier_table(scores):
    """Print out a table displaying scores, number of features or parameter 
    settings for each classifier in the scores dictionary in a separate row.

    Args:
        scores: Dictionary containing informmation about classifiers that should
            be displayed in the table.
    
            {clf_name1: {prop1: v1, prop2: v2, ...},
             clf_name2: {...}, 
             ...}
    """
    
    # get column names
    col_headers = ["Classifier"]
    for k in next(iter(scores.values())).keys():
        col_headers.append(k.title())
    
    # get row data
    rows = []
    for clf in scores.keys():
        row = [round(v, 4) if isinstance(v, float) else v for v in scores[clf].values()]
        row.insert(0, clf)
        rows.append(row)
    
    # find longest string for each column, determines column width
    cols = [list(x) for x in zip(*rows)]
    cols = [[str(x) for x in l1] for l1 in cols]
    
    col_widths = []
    for c in cols:
        col_widths.append(max(len(x) for x in c) + 2)
    
    # check if header itself is longer than longest column value, if so replace 
    # the width value
    col_widths = [w if w >= len(h) + 2 else len(h) + 2 for h, w in zip(col_headers, col_widths)]
    
    # print out results table header
    header_str = ""
    for h, w in zip(col_headers, col_widths):
        header_str += "{col_header: <{width}}".format(col_header=h, width=w)
    
    print(header_str)
    print("-" * len(header_str))
    
    # print out table rows
    for r in rows:
        row_str = ""
        for v, w in zip(r, col_widths):
            row_str += "{val: <{width}}".format(val=str(v), width=w)
        print(row_str)


# classifier names and objects    
names = ['Gaussian NB Classifier', 'Support Vector Classifier', 'KNeighbors Classifier', 
         'Decision Tree Classifier', 'Random Forest Classifier', 'AdaBoost Classifier']
classifiers =[GaussianNB(),
              SVC(random_state=42),
              KNeighborsClassifier(),
              DecisionTreeClassifier(random_state=42),
              RandomForestClassifier(random_state=42),
              AdaBoostClassifier(random_state=42)]

# # create score overview table
# scores = get_multi_classifier_scores(names, classifiers, features, labels, 
#                                      random_state=42)
# # print("Initial screening model perfromances:")
# print_classifier_table(scores)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### feature selection
# # dictionary to store all results
# clf_scores = {}
# for n in names:
#     clf_scores[n] = {"accuracy": [], 'precision': [], 'recall': []}

# # loop trough different k values
# for k in range(1, len(features[1]) + 1):
#     X_k = SelectKBest(k=k).fit_transform(features, labels)
#     scores = get_multi_classifier_scores(names, classifiers, X_k, labels, random_state=42)
    
#     # aggregate the different metrics in the results dictionary for easy plotting
#     # afterwards
#     for n, score in scores.items():
#         accuracy = score['accuracy']
#         precision = score['precision']
#         recall = score['recall']
        
#         clf_scores[n]['accuracy'].append(accuracy)
#         clf_scores[n]['precision'].append(precision)
#         clf_scores[n]['recall'].append(recall)

# # visualize metrics for all algorithms as a function of k
# fig, axs = plt.subplots(2, 3, figsize=(14, 15))
# k = np.arange(1, len(features[1]) + 1)

# for ax, title in zip(axs.flatten(), names):
#     for m in sorted(clf_scores[title].keys()):
#         ax.plot(k, clf_scores[title][m], label=m, marker='o', alpha=0.5)
#         ax.set(xlim=[0.5, 21.5], xticks=k, ylim=[0.0, 1.0], yticks=np.arange(0, 1.1, 0.25))
#         ax.set_xlabel("k", fontsize=10)
#         ax.set_title(title, fontsize=10)
#         ax.tick_params(axis='both', which='major', labelsize=8, direction='in', pad=1)
#         ax.grid()
        
# plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.175))
# plt.show()

### parameter tuning


def find_best_parameters(names, classifiers, X, y, param_grid, score='accuracy', random_state=None):
    """Exhaustive search over specified parameter values for passed classifiers 
    optimizing based on the specified scoring metric.
    
    Args:
        names: List of classifier names.
        classifiers: List of scikit-learn classifier objects.
        X: Feature numpy array or DataFrame.
        y: Label numpy array Dataframe.
        param_grid: Dictionary of parameter dictionaries. The keys have to be
            equal to the entries in the names list.
        score: Scoring metric. Can be set to 'accuracy', 'precision' or
            'recall'.
        random_state: Seed for randomizer in the StratifiedShuffleSplit()
            function.

    Returns:
        Dictionary containing dictionairies for each classifier name key with
        the optimally performing parameter set and the corresponding score.

        {clf_name1: {parameters: {p1: v1, p2: v2, ...}, score: ...},
         clf_name2: {...}, 
         ...}
    """
    
    # check if data set is in a dataframe, if so convert it to a numpy array
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    clf_scores = OrderedDict()
    
    for n, clf in zip(names, classifiers):
        clf_scores[n] = OrderedDict()

        cv = StratifiedShuffleSplit(n_splits=100, test_size=0.33, 
                                    random_state=random_state)
        clf = GridSearchCV(clf, param_grid[n], cv=cv, scoring=score)
        clf.fit(X, y)
        
        clf_scores[n]['parameters'] = clf.best_params_
        clf_scores[n][score] = clf.best_score_
    
    return clf_scores


# parameter dictionary
param_grid = {'Support Vector Classifier': {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
              'KNeighbors Classifier': {'n_neighbors': [2, 5, 10, 20],
                                        'weights': ['uniform', 'distance'],
                                        'algorithm': ['ball_tree', 'kd_tree', 'brute']},
              'Decision Tree Classifier': {'criterion': ['gini', 'entropy'], 
                                           'min_samples_split': [2, 5, 10, 20]},
              'Random Forest Classifier': {'n_estimators': [5, 10, 20, 50, 100], 
                                           'criterion': ['gini', 'entropy'], 
                                           'min_samples_split': [2, 5, 10, 20]},
              'AdaBoost Classifier': {'n_estimators': [5, 10, 20, 50, 100]}}

# # optimize precision
# precision_scores = find_best_parameters(names[1:], classifiers[1:], features, labels, param_grid, score='precision', 
#                                         random_state=42)
# print_classifier_table(precision_scores)
# print()

# # optimize recall
# recall_scores = find_best_parameters(names[1:], classifiers[1:], features, labels, param_grid, score='recall', random_state=42)
# print_classifier_table(recall_scores)

### final model selection (takes a long time to run!)


def optimize_k_and_parameters(names, classifiers, X, y, param_grid, score='accuracy', random_state=None):
    """Find the optimum combination of classifier, number of top features to use
    and parameter settings. Optimization is based on the scoring metric passed.

    Args:
        names: List of classifier names.
        classifiers: List of scikit-learn classifier objects.
        X: Feature numpy array or DataFrame.
        y: Label numpy array Dataframe.
        param_grid: Dictionary of parameter dictionaries. The keys have to be
            equal to the entries in the names list.
        score: Scoring metric. Can be set to 'accuracy', 'precision' or
            'recall'.
        random_state: Seed for randomizer in the StratifiedShuffleSplit()
            function.
    Returns:
        Dictionary containing dictionairies for each classifier name key with
        the optimally performing parameter set, the optimum number of top 
        features to use and the corresponding score.

        {clf_name1: {parameters: {p1: v1, p2: v2, ...}, n: ..., score: ...},
         clf_name2: {...}, 
         ...}
    """

    # perform parameter optimization for varying number of input features
    clf_scores = OrderedDict()
    for k in range(1, len(X[1]) + 1):
        X_k = SelectKBest(k=k).fit_transform(X, y)
        
        scores = find_best_parameters(names, classifiers, X_k, y, param_grid, 
                                      score=score, random_state=random_state)
        clf_scores[k] = scores
    
    # select best results for each classifier
    clf_best_scores = OrderedDict()
    for n in names:
        clf_best_scores[n] = OrderedDict()
        best_score, best_k, best_params = 0, 0, None
        
        for i, v in clf_scores.items():
            for k, v in v.items():
                if k == n:
                    if v[score] > best_score:
                        best_score = v[score]
                        best_k = i
                        best_params = v['parameters']
        
        clf_best_scores[n]['k'] = best_k
        clf_best_scores[n]['parameters'] = best_params
        clf_best_scores[n][score] = best_score
            
    return clf_best_scores


# # optimize precision scores by varying parameters and number of top input features
# precision_scores = optimize_k_and_parameters(names[1:], classifiers[1:], features, labels, param_grid, score='precision', 
#                                              random_state=42)
# print_classifier_table(precision_scores)
# print()

# # optimize recall scores by varying parameters and number of top input features
# recall_scores = optimize_k_and_parameters(names[1:], classifiers[1:], features, labels, param_grid, score='recall', 
#                                           random_state=42)
# print_classifier_table(recall_scores)

### find feature names corresponding with k values
# for k in [3, 6]:
#     selector = SelectKBest(k=k)
#     selector.fit(features, labels)

#     #features = X[selector.get_support(indices=True)].columns.values
#     ixs = selector.get_support(indices=True)
#     feature_names = [f for i, f in enumerate(features_list[1:]) if i in ixs]
#     scores = selector.scores_

#     score_list = sorted([(f, s) for f, s in zip(feature_names, scores)], key=lambda tup: tup[1], reverse=True)

#     # print out scoring table
#     print("k = {}".format(k))
#     print("{:<25} {:<10}".format("Feature", "Score"))
#     print("-------------------------------")
#     for tup in score_list: 
#         print("{:<25} {}".format(tup[0], round(tup[1], 2)))
#     print()

### final model performances
# model definition
top_names = ['Gaussian NB Classifier', 'KNeighbors Classifier', 'Random Forest Classifier', 'AdaBoost Classifier']
top_classifiers =[GaussianNB(),
                  KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='ball_tree'),
                  RandomForestClassifier(min_samples_split=2, n_estimators=5, criterion='gini', random_state=42),
                  AdaBoostClassifier(n_estimators=100, random_state=42)]
top_ks = [6, 3, 3, 3]

# evaluate models
scores = []
for clf, k in zip(top_classifiers, top_ks):
    X_k = SelectKBest(k=k).fit_transform(features, labels)
    
    scores.append(get_classifier_scores(clf, X_k, labels, random_state=42))   

# create score table
clf_scores = OrderedDict()
for n, score in zip(top_names, scores):
    clf_scores[n] = OrderedDict()
    clf_scores[n]['accuracy'] = score[0]
    clf_scores[n]['precision'] = score[1]
    clf_scores[n]['recall'] = score[2]

print("Final classifier performances:")
print()
print_classifier_table(clf_scores)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
final_features_list = ['salary', 'bonus', 'exercised_stock_options', 
					  'total_stock_value', 'long_term_incentive', 
					  'deferred_income']
final_clf = GaussianNB()

dump_classifier_and_data(final_clf, my_dataset, final_features_list)