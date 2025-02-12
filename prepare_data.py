import pandas as pd
import random
from datasets import load_dataset

import yaml
import re
from huggingface_hub import login

random.seed(1996)

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

data_params = config.get('data_params')

with open('configs/keys.yaml') as f:
    keys = yaml.safe_load(f)

def format_question(question):

    desired_format = (
        'A conversation between User and Assistant. ' +
        'The User asks a mathematical question, and the Assistant solves it. ' +
        'The Assistant first reasons step by step, showing working, and then provides the user with the answer. ' +
        'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. ' +
        r'The answer should be only LaTeX and enclosed within \boxed.' +
        '\nUser: ' + question +
        '\nAssistant: <think>'
        )

    return desired_format

def format_answer(answer):
    '''
    The answers are not given in a consistent format - this standardises them
    Answers will always be enclosed by single $ and have no trailing whitespaces
    '''

    # Replace $$ by $
    answer = answer.replace('$$', '$')

    # Regex pattern
    pattern = r'\$(.*?)\$'

    # Find the match
    match = re.search(pattern, answer)

    if match:
        # Extract the content between $$
        content = match.group(1)
        final_answer = '$' + content.strip() + '$'
    elif answer.startswith('\\'):
        final_answer = '$' + answer.strip() + '$'
    else:
        final_answer = ''

    if 'boxed' in final_answer:
        return final_answer
    else:
        return ''

if __name__ == '__main__':
    processed_data = pd.read_csv('data/HARDMath.csv', index_col=0)

    if data_params.get('run_clean_data'):
        # The original dataset contains some numerical metrics that are not needed for our purposes
        # This creates duplicates and should be removed 
        processed_data = processed_data[
            ['question', 'solution', 'question_type', 'answer_type',
             'extracted_answer', 'source']].drop_duplicates()
        print('\nData cleaned')

    if data_params.get('run_training_format'):
        processed_data['prompt'] = processed_data['question'].map(format_question)
        processed_data['ground_truths'] = processed_data['extracted_answer'].map(format_answer)
        processed_data = processed_data[processed_data['ground_truths'] != ''] # Drop empty answers
        processed_data.reset_index(drop=True).to_csv('data/HARDMath_processed.csv')
        print('\nData formatted for training')
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
        print('\nData split into train/test')
    if data_params.get('save'):

        login(token=keys.get('huggingface_key'))

        train.to_csv('data/splits/training_dataset.csv')
        val.to_csv('data/splits/validation_dataset.csv')

        train.push_to_hub('HARDMath_processed_training')
        val.push_to_hub('HARDMath_processed_validation')

