# exploration.py

import numpy as np
import pandas as pd
import pickle

from format_data import convert_dict_to_df

# load dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# convert specified features to data frame
features = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 
			'director_fees', 'exercised_stock_options', 'expenses', 
			'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', 
			'restricted_stock_deferred', 'salary', 'total_payments', 
			'total_stock_value', 'email_address', 'from_messages', 
			'from_poi_to_this_person', 'from_this_person_to_poi', 
			'shared_receipt_with_poi', 'to_messages']

data_df = convert_dict_to_df(data_dict, features, remove_NaN=True, 
						remove_all_zeroes=True, remove_any_zeroes=False, 
						sort_keys=True)

print(data_df.shape)
