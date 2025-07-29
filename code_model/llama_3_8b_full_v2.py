import os,sys
import pandas as pd
from time import sleep, time
from datetime import date

import threading

from openai import OpenAI

# get date and api key
today = date.today()

# setup client
client = OpenAI(api_key="838a259fd73856ed418fc3af1ff1f731d5f2129530d32f0b5ae8764c564942a8",
  base_url='https://api.together.xyz',
)


# load data 
df_data = pd.read_csv('../data/poc_revenue_mcap_data_v2.csv')
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
    for index, row in df_copy.iterrows():

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
                    model="meta-llama/Llama-3-8b-chat-hf",
                    messages=prompt_json,
                    temperature=0.0,
                    max_tokens=100
            )
        except Exception as e:
            print(e)
            # i = i - 1
            sleep(10.0)

        answer = chat_completion.choices[0].message.content

        # print(answer)
        
        temp_list = list(row)
        temp_list.append(message)
        temp_list.append(answer)
        
        output_list.append(temp_list)
        sleep(0.4) 

    results = pd.DataFrame(output_list, columns=['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', 'prompt', "prompt_output"])

    time_taken = int((time() - start_t)/60.0)
    results.to_csv(f'../data/llm_prompt_outputs_v2/llama8b_yearly_3/llama_3_8b_year_{year}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv', index=False)

threads = []

# Create and start a thread for each year
for year in range(1980, 2023):
    t = threading.Thread(target=process_year, args=(year, df_data))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()
