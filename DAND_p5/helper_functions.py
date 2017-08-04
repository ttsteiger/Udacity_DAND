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
    
    
def print_score_table(names, classifiers, X, y, random_state=None):
    """
    Print out table containing accuracy, precision and recall scores for the passed classifiers.
    
    Args:
    
    
    """
    # dictionary to store results of all validation runs
    clf_results = {} 
    for n in names:
        clf_results[n] = {'accuracy': [], 'precision': [], 'recall': []}

    # training and test set split
    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.33, random_state=random_state)
    for train_ixs, test_ixs in sss.split(X, y):
        X_train, X_test = X[train_ixs, :], X[test_ixs, :]
        y_train, y_test = y[train_ixs] , y[test_ixs]

        # loop trough all classifiers
        for n, clf in zip(names, classifiers):      
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)

            accuracy = clf.score(X_test, y_test)
            precision = precision_score(y_test, pred)
            recall = recall_score(y_test, pred)

            # store scores in the respective result list
            clf_results[n]['accuracy'].append(accuracy)
            clf_results[n]['precision'].append(precision)
            clf_results[n]['recall'].append(recall)

    # print out results
    print("{:<25} {:<10} {:<10} {}".format("Classifier", "Accuracy", "Precision", "Recall"))
    print("------------------------------------------------------")
    for n in names:
        accuracy = round(np.mean(clf_results[n]['accuracy']), 2)
        precision = round(np.mean(clf_results[n]['precision']), 2)
        recall = round(np.mean(clf_results[n]['recall']), 2)

        print("{:<25} {:<10} {:<10} {}".format(n, accuracy, precision, recall))


def best_parameter_search(names, classifiers, X, y, param_grid, score='accuracy', random_state=None):
    """
    Exhaustive search over specified parameter values for passed classifiers. Prints out a table 
    displaying the results.
    
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

    print("{:<25} {:<10} {}".format("Classifier", score.title(), "Parameters"))
    print("------------------------------------------------")
    for n, clf in zip(names, classifiers):
        clf_scores_parameters[n] = {}

        cv = StratifiedShuffleSplit(n_splits=100, test_size=0.33, random_state=random_state)
        clf = GridSearchCV(clf, param_grid[n], cv=cv, scoring=score) #'{}_macro'.format(
        clf.fit(X, y)

        print("{:<25} {:<10} {}".format(n, round(clf.best_score_, 2), clf.best_params_))