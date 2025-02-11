import numpy as np
import pandas as pd
import random
from datasets import load_dataset

import yaml

random.seed(1996)

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

data_params = config.get('data_params')

if __name__ == '__main__':
    if data_params.get('run_clean_data'):
        # The original dataset contains some numerical metrics that are not needed for our purposes
        # This creates duplicates and should be removed 
        processed_data = pd.read_csv('data/HARDMath.csv', index_col=0)
        processed_data = processed_data[
            ['question', 'solution', 'question_type',
              'answer_type', 'extracted_answer', 'source']].drop_duplicates()
        processed_data.to_csv('data/HARDMath_processed.csv')

    if data_params.get('run_split'):
        # Load .csv file as a Dataset object
        fulldata = (
            load_dataset('csv', data_files='data/HARDMath_processed.csv')
            .get('train'))
        # Split into train/validation
        # For testing we use the evaluation/data/HARDMath_mini.csv
        train, val = (
            fulldata
            .train_test_split(data_params.get('split_size'))
            .values())

    if data_params.get('save'):
        train.to_csv('data/splits/training_dataset.csv')
        val.to_csv('data/splits/validation_dataset.csv')

