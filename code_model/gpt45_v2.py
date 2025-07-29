import os,sys
import pandas as pd
from time import sleep, time
from datetime import date

import threading

from openai import OpenAI
sys.path.insert(0, '/home/research/git repos/zero-shot-finance')
from api_keys import APIKeyConstants

client = OpenAI(api_key=APIKeyConstants.OPENAI_API_KEY)

# get date and api key
today = date.today()


# load data 
df_data = pd.read_csv('../data/sampled_data_v2.csv')
df_data = df_data[['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK']]



def process_year(year, df_data):

    df_copy = df_data.copy()

    df_copy = df_copy.loc[(df_copy['year'] == year)]
    df_copy = df_copy.reset_index(drop=True)
    # df_copy = df_copy.head(10)
    print(df_copy.shape)
    print(df_copy.head())

    start_t = time()

    output_list = []
    index = 0
    while index < len(df_copy): 
        row = df_copy.loc[index]

        if index%20 == 0:
            print(index)
        
        company_name = row['CONM']
        financial_year = str(row['year'])
        message = f'What was the revenue of {company_name} in financial year {financial_year}?'

        prompt_json = [
                {"role": "user", "content": message},
        ]
        try:
            chat_completion = client.chat.completions.create(
                    model="gpt-4.5-preview-2025-02-27",
                    messages=prompt_json,
                    temperature=0.00,
                    max_tokens=100
            )
        except Exception as e:
            print(e)
            index = index - 1
            sleep(10.0)

        answer = chat_completion.choices[0].message.content

        # print(answer)
        
        temp_list = list(row)
        temp_list.append(message)
        temp_list.append(answer)
        
        output_list.append(temp_list)
        sleep(0.1) 
        index += 1

    results = pd.DataFrame(output_list, columns=['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', 'prompt', "prompt_output"])

    time_taken = int((time() - start_t)/60.0)
    results.to_csv(f'../data/llm_prompt_outputs_v2/gpt45_yearly/gpt4o_year_{year}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv', index=False)

threads = []

# Create and start a thread for each year
for year in range(1980, 2023):
    t = threading.Thread(target=process_year, args=(year, df_data))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()