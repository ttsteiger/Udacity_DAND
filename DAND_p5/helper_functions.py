# helper_functions.py
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler


# data wrangling
################################################################################


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
        Pandas DataFrame with each row representing a data point. The column 
        names are equal to the features passed.
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
        # first entry of data point is the name of the person
        val_dict = {'name': key}

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
            	# exclude 'poi' and 'name' from criteria
                if key != 'poi' and key != 'name':
                    if val != 0 and val != "NaN":
                        append = True
                        break
        
        # don not add single zero data points if remove_any_zeroes is set to 
        # True
        elif remove_any_zeroes:
            append = True
            # exclude 'poi' and 'name' from criteria
            keys =  [f for f in features if f not in ('poi', 'name')]
            # list containing values of remaining features
            val_list = [val_dict.get(k) for k in keys]

            if 0 in val_list or "NaN" in val_list:
                append = False
        
        # all data points are added 
        else:
            append = True
    
        
        # append data point if it is flagged for addition
        if append:
            df = df.append(val_dict, ignore_index=True)
        
    return df
   
# print out classifier table
################################################################################


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

# plotting
################################################################################


def scatter_plot(df, x, y, normalize=True):
    """Create a scatter plot of column x vs. columns y. Also normalizes the two
    variables using the MinMaxScaler() from scikit-learn.

    Args:
        df: DataFrame containing the feature columns.
        x: Header string of the x column. 
        y: Header string of the y column.
        normalize: Enable or disable feature normalization.
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
    ax.set(title="{} vs. {}".format(x, y), 
           xlabel=x, xlim=[-0.02, 1.02], 
           ylabel=y, ylim=[-0.02, 1.02])
    plt.legend()
    
    plt.show()

    
def generate_meshgrid(x, y, h=.01):
    """Source: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
    
    Create a meshgrid of points to plot in.
    
    Args:
        x: 1D numpy array to base x-axis meshgrid on.
        y: 1D numpy array to base y-axis meshgrid on.
        h: Stepsize for meshgrid.

    Returns:
        xx, yy: 1D arrays containing x- and y-axis meshgrids.
    """

    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_boundaries(ax, clf, xx, yy, **params):
    """Source: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
    
    Plot the decision boundaries for a classifier.

    Args:
        ax: Matplotlib axes object.
        clf: Scikit-learn classifier object.
        xx: 1D array containing x-axis meshgrid.
        yy: 1D array containing y-axis meshgrid.
        params: Dictionary of parameters to pass to contourf function.
    """
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    
    return out