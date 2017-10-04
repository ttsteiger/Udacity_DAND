# data_preparation.py

import numpy as np
import pandas as pd

# load original dataset
data_df = pd.read_csv('earthquakes.csv')

# add magnitude category column
bin_labels = ['Moderate', 'Strong', 'Major', 'Great']
bins = [data_df['Magnitude'].min(), 5.9, 6.9, 7.9, data_df['Depth'].max()]
data_df['Magnitude Cat'] = pd.cut(data_df['Magnitude'], bins, labels=bin_labels)

# create bins for depth variable
bin_labels = ['Shallow', 'Intermediate', 'Deep']
bins = [data_df['Depth'].min(), 69.9, 299.9, data_df['Depth'].max()]
data_df['Depth Cat'] = pd.cut(data_df['Depth'], bins, labels=bin_labels)

print(data_df.shape)

# store df as .csv
data_df.to_csv('earthquakes_edited.csv', index=False)