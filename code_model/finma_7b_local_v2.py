import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM

import pandas as pd
import numpy as np
from time import time
from datetime import date


today = date.today()


# load data 
df_data = pd.read_csv('../data/sampled_data_v2.csv')
df_data = df_data[['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK']]


# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str("0")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device assigned: ", device)

# get model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained('TheFinAI/finma-7b-full')
model = LlamaForCausalLM.from_pretrained('TheFinAI/finma-7b-full', device_map='auto')

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


for year in range(2022, 1979, -1):

    df_copy = df_data.copy()

    df_copy = df_copy.loc[(df_copy['year'] == year)]
    df_copy = df_copy.reset_index(drop=True)
    # df_copy = df_copy.head(10)
    print(df_copy.shape)
    print(df_copy.head())
    



    prompts_list = []
    for index, row in df_copy.iterrows():
        
        company_name = row['CONM']
        financial_year = year
        message = f'What was the revenue of {company_name} in financial year {financial_year}?'
        prompts_list.append(message)

    
    start_t = time()

    # documentation: https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation
    res = pipeline(
        prompts_list, 
        max_new_tokens=100, 
        do_sample=False, 
        use_cache=True, 
        temperature=0.00, 
        eos_token_id=tokenizer.eos_token_id
        )
    
    output_list = []
    
    for index, row in df_copy.iterrows():
        answer = res[index][0]['generated_text'][len(prompts_list[index]):]
        answer = answer.strip()
        print(answer)

        temp_list = list(row)
        temp_list.append(prompts_list[index])
        temp_list.append(answer)

        output_list.append(temp_list)

    results = pd.DataFrame(output_list, columns=['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', 'prompt', "prompt_output"])

    time_taken = int((time() - start_t)/60.0)
    results.to_csv(f'../data/llm_prompt_outputs_v2/finma_7b_yearly/finma_7b_chat_year_{year}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv', index=False)