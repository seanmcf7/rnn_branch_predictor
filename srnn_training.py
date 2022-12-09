import pandas as pd
import numpy as np
from srnn import SRNN

def main():
    train_df = pd.read_csv('cleaned_datasets/full/I04.csv', dtype=np.int8)

    srnn = SRNN(pht_dtype=np.int32, pht_size=256, pht_update_weight=3)

    correct_predictions = 0
    total_predictions = 0
    for index, row in train_df.iterrows():
        prediction = srnn.predict(row['PC'])
        if prediction == row['result']:
            correct_predictions += 1
        total_predictions += 1
        
        srnn.update_pht(pc=row['PC'], predicted=prediction, actual=row['result'])
        srnn.update_ght(1 if row['results'] == 1 else -1)

        print('{:<8d}: Accuracy={:.4f} {}'.format(total_predictions, correct_predictions / total_predictions, 'O' if prediction == row['result'] else 'X'))

    print(f'Total correct predictions: {correct_predictions}/{total_predictions}')

if __name__ == '__main__':
    main()