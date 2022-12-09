import multiprocessing
import numpy as np
import os
import pandas as pd
from srnn import SRNN

def main():
    dataset_path = 'cleaned_datasets/full'
    datasets = ['I04', 'INT03', 'MM03', 'MM05', 'S02', 'S04']

    processes = []

    for dataset in datasets:
        processes.append(multiprocessing.Process(target=train_srnn, args=(dataset,dataset_path,)))
    
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()


def train_srnn(dataset, path):
    output_path = f'results_v4_dec8/{dataset}'

    pht_dtypes = [np.int8, np.int16, np.int32, np.int64]
    pht_sizes = [64, 128, 256, 512, 1024, 4096]
    training_lengths = [10000, 100000, 400000]
    weights = [1, 3, 5]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    columns = ['PHT Dtype', 'PHT Size', 'Training Length', 'Update Weights']
    record_accuracies = [10000, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000]
    acc_labels = [f'Acc: {a}' for a in record_accuracies]
    columns.extend(acc_labels)

    out_file = f'{output_path}/{dataset}_results.csv'
    results_list = []
    input_df = pd.read_csv(f'{path}/{dataset}.csv')
    
    for type in pht_dtypes:
        for size in pht_sizes:
            print(f'{dataset}: Starting PHT {size} {type(0).dtype}')
            for training in training_lengths:
                for weight in weights:
                    results_dict = dict.fromkeys(columns)
                    results_dict['PHT Dtype'] = type(0).dtype
                    results_dict['PHT Size'] = size
                    results_dict['Training Length'] = training
                    results_dict['Update Weights'] = weight

                    srnn = SRNN(pht_dtype=type, pht_size=size, pht_update_weight=weight)

                    correct_predictions = 0
                    total_predictions = 0
                    for index, row in input_df.iterrows():
                        prediction = srnn.predict(row['PC'])
                        if prediction == row['result']:
                            correct_predictions += 1
                        total_predictions += 1
                        if (index < training):
                            srnn.update_pht(pc=row['PC'], predicted=prediction, actual=row['result'])
                        srnn.update_ght(1 if row['result'] == 1 else -1)
                        
                        if total_predictions in record_accuracies:
                            results_dict[f'Acc: {total_predictions}'] = correct_predictions / total_predictions

                    results_list.append(results_dict)
    pd.DataFrame.from_dict(results_list).to_csv(out_file)


if __name__ == '__main__':
    main()
    
