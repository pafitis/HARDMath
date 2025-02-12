
import re
import yaml

from Levenshtein import distance

## Config
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

rl_params = config.get('rl_params')

## Reward functions
def correctness_reward(completion, ground_truth):
    
    prediction = completion[0]['content']
    
    if prediction is None:
        return 0.0
    else:
        # Extract the contents between <answer> ... </answer>
        start_point = prediction.find('<answer>') + len('<answer>')
        end_point = prediction.find('</answer>')
        final_prediction = prediction[start_point:end_point]

        # TODO: might need to remove ',' as found in list answers

        # Will use Levenshtein distance (edit distance)
        # counts number of additions/removals/edits required
        # to match the desired string

        lev_distance = distance(final_prediction, ground_truth) # symmetric, order doesn't matter
        len_maxstring = max(len(final_prediction), len(ground_truth))

        # TODO:
        # I've observed that we should probably modify the distance
        # We should probably have 0 reward if required modifications are more than 4-5
        # and then normalise based on that?
        return 1.0 - (lev_distance / len_maxstring)

def correctness_reward_wrapper(completions, ground_truth, **kwargs):
    '''
    Checks the correctness of the prediction against the ground truth
    Uses Levenshtein distance (edit distance)
    
    TODO: modify distance metric
          currently feels like its not penalising very wrong answers enough

    completions: output from model
    ground_truth: sourced directly from HARDMath (column: extracted_answer)
    **kwargs: requirement from GRPOTrainer, see docs: https://huggingface.co/docs/trl/main/en/grpo_trainer
    '''
    return [correctness_reward(completion, truth) for completion, truth in zip(completions, ground_truth)]

def formating_reward(completion, eos_token = '<|eot_id|>'):
    '''
    The desired format should contain two main blocks
        <think> ... </think>
        \n
        <answer> $$\boxed{...}$$ </answer>

    Thoughts:
        1. The final answer should use \\boxed LaTeX
        2. The final answer should be enclosed within <answer> </answer>    
        3. There should be no \\boxed string within <think> </think>
        4. EOS should appear after </answer>

    Note:
        Compared re, .find, `in` approaches 
        `in` appears fastest, however, `in` requires at least 3 checks which in turn
        makes it slower than a single regex search

        Missing <> should be captured, as well as wrong order
        No extra chars before or after
    '''
    # Checks general formatting
    # Check if starts with <think> and ends with </answer>
    
    # Checks that \boxed does not appear in <think> </think>
        # (?:(?!\\boxed\{).)*
    # Checks that $\boxed{...} appears in <answer> </answer>
    # incl. whitepsaces to match HARDMath format
        # \$\$\\boxed.*\}\$\$
    # Checks that thinking and answering parts are sep by new line
        # </think>\\n<answer>
    
    prediction = completion[0]['content']
    
    pattern = (
        r'^<think>(?:(?!\\boxed\{).)*</think>\n<answer>\$\\boxed.*\}\$</answer>$'
    )
    
    return rl_params.get('reward_format') if re.match(pattern, prediction, re.DOTALL) else 0.0

def formating_reward_wrapper(completions, **kwargs):
    return [formating_reward(completion) for completion in completions]

# TODO: think about reward function that promotes lengthy responses?