import os,sys
import pandas as pd
from time import sleep, time
from datetime import date

import google.generativeai as genai

import threading


google_api_key = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=google_api_key)

def get_safety_settings():
    """ 
    Set the block threshold to None or whatever you think best for each harm category.
    **If you want to block most threatening things, use BLOCK_LOW_AND_ABOVE for threshold**
    **If you want to block almost nothing, use BLOCK_NONE for threshold**
    Refer https://ai.google.dev/gemini-api/docs/safety-settings to modify the safety settings
    Gemini model information: https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro
    
    For sensitive work, this is very important. 
    """
    safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_LOW_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_LOW_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_LOW_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_LOW_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_LOW_AND_ABOVE",
    },
    ]
    return safety_settings

model_name = 'gemini-1.5-pro' #change 

safety_settings = get_safety_settings()

model = genai.GenerativeModel(model_name = model_name, \
                              safety_settings = safety_settings,
                              generation_config=genai.GenerationConfig(\
                              max_output_tokens=100,\
                              temperature=0.0)
                             )

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
    for index, row in df_copy.iterrows():
        
        company_name = row['CONM']
        financial_year = str(row['year'])
        message = f'What was the revenue of {company_name} in financial year {financial_year}?'

        prompt_json = [
                {"role": "user", "parts": [message]},
        ]
        try:
            chat_completion = model.generate_content(prompt_json)
        except Exception as e:
            print(e)
            # i = i - 1
            sleep(10.0)
        try: 
            answer = chat_completion._result.candidates[0].content.parts[0].text
        except Exception as e:
            answer = "Error in Gemini"    
        print(answer)
        
        temp_list = list(row)
        temp_list.append(message)
        temp_list.append(answer)
        
        output_list.append(temp_list)
        sleep(0.5) 

    results = pd.DataFrame(output_list, columns=['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', 'prompt', "prompt_output"])

    time_taken = int((time() - start_t)/60.0)
    results.to_csv(f'../data/llm_prompt_outputs_v2/gemini15_yearly_small/gemini_15_pro_year_{year}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv', index=False)

threads = []

# Create and start a thread for each year
for year in range(2012, 2023):
    t = threading.Thread(target=process_year, args=(year, df_data))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()