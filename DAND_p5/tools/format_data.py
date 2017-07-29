# format_data.py

import numpy as np
import pandas as pd


def convert_dict_to_arr(dictionary, features, remove_NaN=True, 
                        remove_all_zeroes=True, remove_any_zeroes=False, 
                        sort_keys=False):
    """ 
    Convert dictionary to a numpy array of features.

    Args:
        dictionary: Dictionary containing the feature names as keys and the 
            corresponding values.
        features: List of feature names. First feature passed needs to be 'poi'.
        remove_NaN: True converts all "NaN" strings to 0.
        remove_all_zeroes: True omits all 0 data points.
        remove_any_zeroes: True omits single 0 data points.
        sort_keys: True sorts the dictionary keys in alphabetical order before
            adding the data points to the numpy array.

    Returns:
        Function returns a numpy array with each row representing a data point
        with the specified features in its columns. All returned values are
        strings since numeric and string variables are present for all variables
        but numpy arrays can only contain a single data type.
    """

    # check that first feature passed is 'poi'
    assert (features[0] == 'poi'), "The first feature needs to be 'poi'!"

    # list to store the data points as numpy arrays
    return_list = []

    # sort keys alphabetically if sort_keys is set to True
    if sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    # loop trough the data dictionary 
    for key in keys:
        
        val_list = [key] # first entry of data point is the name of the person

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
                val = np.Nan
            # set NaN values to 0 if remove_NaN is set to True
            elif val == "NaN" and remove_NaN:
                val = 0
            val_list.append(val)

        # do not add all zero data points if remove_all_zeroes is set to True
        if remove_all_zeroes:
            append = False
            for val in val_list[1:]: # exclude 'poi' from criteria
                if val != 0 and val != "NaN":
                    append = True
                    break
        
        # don not add single zero data points if remove_any_zeroes is set to 
        # True
        elif remove_any_zeroes:
            append = True
            if 0 in val_list[1:] or "NaN" in val_list[1:]:
                append = False
        
        # all data points are added 
        else:
            append = True

        # append data point if it is flagged for addition
        if append:
            return_list.append(np.array(val_list, dtype=str))

    return np.array(return_list)


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