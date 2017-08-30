# helper_functions.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler


def convert_dict_to_df(dictionary, features, remove_NaN=True, 
                        remove_all_zeroes=True, remove_any_zeroes=False, 
                        sort_keys=False):
    """
    Convert dictionary to a pandas data frame of features.
    
    Args:
        dictionary: Dictionary containing the feature names as keys and the 
            corresponding values.
        features: List of feature names. First feature passed needs to be 'poi'.
        remove_NaN: True converts all "NaN" strings to 0.
        remove_all_zeroes: True omits all 0 data points.
        remove_any_zeroes: True omits single 0 data points.
        sort_keys: True sorts the dictionary keys in alphabetical order before
            adding the data points to the data frame.

    Returns:
        Function returns a pandas data frame with each row representing a data 
        point with the specified features in its columns.
    """

    # check that first feature passed is 'poi'
    assert (features[0] == 'poi'), "The first feature needs to be 'poi'!"

    # data frame to store the data points as individual rows
    df = pd.DataFrame(columns=['name'] + features)

    # sort keys alphabetically if sort_keys is set to True
    if sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    # loop trough the data dictionary 
    for key in keys:
        
        val_dict = {'name': key} # first entry of data point is the name of the person

        for feature in features:
            # check if specified feature exists, throw a warning if not and 
            # stop the function
            try:
                val = dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return

            val = dictionary[key][feature]

            # set 'NaN' strings to np.NaN values
            if val == "NaN" and not remove_NaN:
                val = np.NaN
            # set NaN values to 0 if remove_NaN is set to True
            elif val == "NaN" and remove_NaN:
                val = 0

            val_dict[feature] = val

        # do not add all zero data points if remove_all_zeroes is set to True
        if remove_all_zeroes:       
            append = False
            for key, val in val_dict.items(): 
                if key != 'poi' and key != 'name': # exclude 'poi' and 'name' from criteria
                    if val != 0 and val != "NaN":
                        append = True
                        break
        
        # don not add single zero data points if remove_any_zeroes is set to 
        # True
        elif remove_any_zeroes:
            append = True
            keys =  [f for f in features if f not in ('poi', 'name')] # exclude 'poi' and 'name' from criteria
            val_list = [val_dict.get(k) for k in keys] # list containing values of remaining features

            if 0 in val_list or "NaN" in val_list:
                append = False
        
        # all data points are added 
        else:
            append = True
    
        
        # append data point if it is flagged for addition
        if append:
            df = df.append(val_dict, ignore_index=True)
        
    return df


def scatter_plot(df, x, y, normalize=True):
    """
    
    """
    poi_df = df[df['poi'] == True]
    x_poi = poi_df[x].fillna(value=0).values.reshape(-1, 1)
    y_poi = poi_df[y].fillna(value=0).values.reshape(-1, 1)
    
    non_poi_df = df[df['poi'] == False]
    x_non_poi = non_poi_df[x].fillna(value=0).values.reshape(-1, 1)
    y_non_poi = non_poi_df[y].fillna(value=0).values.reshape(-1, 1)
    
    if normalize:
        x_poi = MinMaxScaler().fit_transform(x_poi)
        y_poi = MinMaxScaler().fit_transform(y_poi)
        
        x_non_poi = MinMaxScaler().fit_transform(x_non_poi)
        y_non_poi = MinMaxScaler().fit_transform(y_non_poi)
    
    # create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(x_poi, y_poi, color="red", label="poi")
    ax.scatter(x_non_poi, y_non_poi, color="blue", label="non poi")
    ax.set(title="{} vs. {}".format(x, y), xlabel=x, xlim=[-0.02, 1.02], ylabel=y, ylim=[-0.02, 1.02])
    plt.legend()
    
    plt.show()
    
    
def get_classifier_scores(clf, X, y, random_state=None):
    """
    
    """
    
    accuracies, precisions, recalls = [], [], []
    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.33, random_state=42)
    
    for train_ixs, test_ixs in sss.split(X.values, y.values):
        X_train, X_test = X.values[train_ixs, :], X.values[test_ixs, :]
        y_train, y_test = y.values[train_ixs] , y.values[test_ixs]
    
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        accuracies.append(clf.score(X_test, y_test))
        precisions.append(precision_score(y_test, pred))
        recalls.append(recall_score(y_test, pred))
    
    return [np.mean(accuracies), np.mean(precisions), np.mean(recalls)]


def get_multi_classifier_scores(names, classifiers, X, y, random_state=None):
    """
    """
    
    clf_results = {} 
    
    for n, clf in zip(names, classifiers):
        clf_results[n] = {}
        scores = get_classifier_scores(clf, X, y, random_state=random_state)
        
        clf_results[n]['accuracy'] = scores[0]
        clf_results[n]['precision'] = scores[1]
        clf_results[n]['recall'] = scores[2]
    
    return clf_results


def find_best_parameters(names, classifiers, X, y, param_grid, score='accuracy', random_state=None):
    """
    Exhaustive search over specified parameter values for passed classifiers optimizing based on the specified
    scoring metric.
    
    Args:
        names:
        classifiers:
        X:
        y:
        param_grid:
        score:
        random_state:

    Returns:
        
    """
    
    clf_scores = {}

    for n, clf in zip(names, classifiers):
        clf_scores[n] = {}

        cv = StratifiedShuffleSplit(n_splits=100, test_size=0.33, random_state=random_state)
        clf = GridSearchCV(clf, param_grid[n], cv=cv, scoring=score)
        clf.fit(X, y)
        
        clf_scores[n][score] = clf.best_score_
        clf_scores[n]['parameters'] = clf.best_params_
    
    return clf_scores


def optimize_features_and_parameters(names, classifiers, X, y, top_features, param_grid, score='accuracy', random_state=None):
    """
    """
    # perform parameter optimization for varying number of input features
    clf_scores = {}
    for i in range(1, len(top_features) + 1):
        features = top_features[:i]
        X_i = X.loc[:, features].values
        
        scores = find_best_parameters(names, classifiers, X_i, y, param_grid, score=score, random_state=random_state)
        clf_scores[i] = scores
    
    # select best results for each classifier
    clf_best_scores = {}
    for n in names:
        best_score, best_i, best_params = 0, 0, None
        
        for i, v in clf_scores.items():
            for k, v in v.items():
                if k == n:
                    if v[score] > best_score:
                        best_score = v[score]
                        best_i = i
                        best_params = v['parameters']
        
        clf_best_scores[n] = {score: best_score, 'input features': best_i, 'parameters': best_params}
            
    return clf_best_scores


def print_classifier_table(scores):
    """
    
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
    
    # check if header itself is longer than longest column value, if so replace the width value
    col_widths = [w if w >= len(h) + 2 else len(h) + 2 for h, w in zip(col_headers, col_widths)] # if/else list comprehension
    
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