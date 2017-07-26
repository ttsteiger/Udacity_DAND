#!/usr/bin/python

import numpy as np
import pandas as pd
import pickle

from feature_format import featureFormat, targetFeatureSplit
from format_data import convert_dict
from tester import dump_classifier_and_data

# load dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

"""
Structure data_dict:
{'NAME': {'poi': ...
          
          'bonus': ...,
          'deferral_payments': ...,
          'deferred_income': ...,
          'director_fees': ...,
          'exercised_stock_options': ...,
          'expenses': ...,
          'loan_advances': ...,
          'long_term_incentive': ...,
          'other': ...,
          'restricted_stock': ...,
          'restricted_stock_deferred': ...,
          'salary': ...,
          'total_payments': ...,
          'total_stock_value': ...,

          'email_address': ...,
          'from_messages': ...,
          'from_poi_to_this_person': ...,
          'from_this_person_to_poi': ...,
          'shared_receipt_with_poi': ...,
          'to_messages': ...
          }
}
"""

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 
			'director_fees', 'exercised_stock_options', 'expenses', 
			'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', 
			'restricted_stock_deferred', 'salary', 'total_payments', 
			'total_stock_value', 'email_address', 'from_messages', 
			'from_poi_to_this_person', 'from_this_person_to_poi', 
			'shared_receipt_with_poi', 'to_messages']

# convert data_dict to a numpy array containing the specified features
#data = convert_dict(data_dict, features, remove_NaN=True,
#	                remove_all_zeroes=True, remove_any_zeroes=False, 
#	                sort_keys=True)

df = pd.DataFrame(columns=features)

print(df)

#print(list(sorted(data_dict.keys()))[:5])
#print(data_dict["ALLEN PHILLIP K"])
#print(data_dict["BADUM JAMES P"])
#print(data_dict["BANNANTINE JAMES M"])
#print()

#print(data.size)
#print(data.shape)
#print(data[:5, :])

"""
### Task 2: Remove outliers

# drop 'TOTAL' row

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

"""