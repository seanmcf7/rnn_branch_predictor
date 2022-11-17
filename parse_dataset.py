import numpy as np
import os
import pandas as pd

# nrows_read = 1000
training_split = 0.8

datasets_path = 'dataset/archive/'
datasets = {'I04'}#, 'INT03', 'MM03', 'MM05', 'S02', 'S04'}

train_output_path = 'train/'
test_output_path = 'test/'

if not os.path.exists(train_output_path):
    os.makedirs(train_output_path)
if not os.path.exists(test_output_path):
    os.makedirs(test_output_path)

for dataset in datasets:
    print(f'Formatting {dataset} dataset...')
    df = pd.read_csv(datasets_path + dataset + '.csv', index_col=0)#, nrows=nrows_read)

    total_datapoints = df.shape[0]

    training_datapoints = int(total_datapoints * training_split)

    # Get PC and result in revised formatting
    df_select = df.iloc[:, len(df.columns)-1].astype('uint8').to_frame(name='result')
    df_select.insert(0, 'PC', 0)
    df_select.insert(1, 'global_history', 0)
    df_select = df_select.astype({'global_history': 'uint64'})
    for i in range(0, 64):
        df_select.insert(2 + i, f'2bit_count{i}', 0)

    sat_count_offset = 33
    for index, row in df.iterrows():
        if (index % 20000 == 0):
            print(index)
        df_select.loc[index, 'PC'] = int("".join(str(int(x)) for x in row[0:32].tolist()), 2)

        # Counters normalized from [0 1], convert to int
        # need to map 0.0 --> 0, 0.33 --> 1, 0.66 --> 2, 1.0 --> 3
        for i in range(0, 64):
            norm_val = row[sat_count_offset + i]
            if norm_val >= 0.8:
                bit_counter = 3
            elif 0.5 <= norm_val < 0.8:
                bit_counter = 2
            elif 0.2 <= norm_val < 0.5:
                bit_counter = 1
            else:
                bit_counter = 0
            df_select.loc[index, f'2bit_count{i}'] = bit_counter

        # Record global history
        if index != 0:
            df_select.loc[index, 'global_history'] = (df_select.loc[index-1, 'global_history'] << np.uint64(1)) | df_select.loc[index-1, 'result']

    print(df_select.dtypes)
    df_select.iloc[:training_datapoints].to_csv(train_output_path + dataset + '_train.csv')
    df_select.iloc[training_datapoints:].to_csv(test_output_path + dataset + '_test.csv')

