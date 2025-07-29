import re
import pandas as pd
import numpy as np
import math
import os


def extract_revenue(s):
    s = str(s).lower()
    style = 0
    matches = [match.group() for match in re.finditer(r'(?<=\$)\d+(\.\d{1,2})?\s*(million|billion|trillion)?', s)]
    if len(matches) != 0:
        style = 1
    else:
        matches = [match.group() for match in re.finditer(r'^\$([\d,]*\.?\d+)([mb])$', s)]
        if len(matches) != 0:
            style = 2
    if style == 0:
        return np.nan    
    elif style == 1:
        string_split = matches[0].split()
        if len(string_split) > 1 and ('million' in string_split[1]):
            return float(string_split[0])
        elif  len(string_split) > 1 and 'billion' in string_split[1]:
            return float(string_split[0])*1000
        else:
            return float(string_split[0])
    elif style == 2:
        if matches[0].group(2) == "m":
            return float(matches[0].group(1))
        elif matches[0].group(2) == "b":
            return float(matches[0].group(1))*1000
        else:
            return float(matches[0].group(1))


def calculate_ad_hoc_error(true_rev, extracted_rev, success):
    if success == False:
        return 100.0
    else:
        if true_rev==0.0:
            return 100.0
        else:
            return ((extracted_rev-true_rev)/true_rev)*100
        

def bool_high(value):
    if abs(value) >= 10.0:
        return True
    else: 
        return False

def bool_low(value):
    if abs(value) < 10.0:
        return True
    else:
        return False
    
def bool_low_5(value):
    if abs(value) < 5.0:
        return True
    else:
        return False

def bool_low_20(value):
    if abs(value) < 20.0:
        return True
    else:
        return False

def combine_files(cols, directory_name, csv_name):
    combined_prompt_outputs = pd.DataFrame(columns=cols)
    for file in os.listdir(directory_name):
        file_path = os.path.join(directory_name, file)
        print(file_path)
        temp_raw_result = pd.read_csv(file_path)
        combined_prompt_outputs = combined_prompt_outputs.append(temp_raw_result)
    combined_prompt_outputs.to_csv(csv_name, index=False)

def calculate_std(x, model):
    correctness = (x['low'] == True).sum() / len(x) * 100
    std = np.sqrt(correctness * (100.0 - correctness) / 100.0)
    return pd.Series([correctness, std], index=[f'{model}_correctness', f'{model}_std'])

def calculate_std_per(x, model):
    correctness = (x['low'] == True).sum() / len(x) * 100
    std = np.sqrt(correctness * (100.0 - correctness) / 100.0)

    correctness_5 = (x['low_5'] == True).sum() / len(x) * 100
    std_5 = np.sqrt(correctness_5 * (100.0 - correctness_5) / 100.0)

    correctness_20 = (x['low_20'] == True).sum() / len(x) * 100
    std_20 = np.sqrt(correctness_20 * (100.0 - correctness_20) / 100.0)

    return pd.Series([correctness, std, correctness_5, std_5, correctness_20, std_20], index=[f'{model}_correctness_10', f'{model}_std_10', f'{model}_correctness_5', f'{model}_std_5', f'{model}_correctness_20', f'{model}_std_20'])

def calculate_std_hal(x, model):
    # hallucination = ((x['success'] == True) & (x['low'] == False)).sum() / len(x) * 100
    hallucination = ((x['success'] == True) & (x['low'] == False)).sum() / (x['low'] == False).sum() * 100
    std = np.sqrt(hallucination * (100.0 - hallucination) / 100.0)
    return pd.Series([hallucination, std], index=[f'{model}_hallucination', f'{model}_std'])

def get_same_comp(df, start_year, end_year):
    years = set(range(start_year, end_year+1))
    filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    same_comp_df = filtered_df.groupby('GVKEY').filter(lambda x: set(x['year']) == years)

    return same_comp_df
    

def extract_recommedation(action_response):

    if "BUY" in action_response:
        return "BUY" 
    # elif ("BUY" in answer and closing[1]>closing[0]) or ("SELL" in answer and closing[1]<=closing[0]):
    elif "SELL" in action_response:
        return "SELL" 
    else:
        return "DNK"

