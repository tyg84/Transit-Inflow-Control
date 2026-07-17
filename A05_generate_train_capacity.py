import pandas as pd
import numpy as np
import _constant


case_name = 'reference'

train_capacity_df = pd.read_csv('data/manual_input_data/train_capacity.csv')
train_capacity_df['train_capacity'] = np.round(
    train_capacity_df['train_capacity'] * _constant.TRAIN_CAPACITY_FACTOR)
train_capacity_df['train_capacity'] = train_capacity_df['train_capacity'].astype('int')

train_capacity_df.to_csv(f'data/{case_name}/train_capacity_adjusted.csv', index=False)
