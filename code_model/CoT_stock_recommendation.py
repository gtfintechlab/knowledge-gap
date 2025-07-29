import pandas as pd
import os,sys

from openai import OpenAI
sys.path.insert(0, '/home/research/git repos/zero-shot-finance')
from api_keys import APIKeyConstants

client = OpenAI(api_key=APIKeyConstants.OPENAI_API_KEY)

from time import sleep

def get_filtered_comanpy():
    model = 'gpt4o'
    df = pd.read_csv(f'../data/llm_prompt_outputs_v2/{model}_yearly_combined.csv')
    df = df.drop(columns=['prompt_output'], errors='ignore')
    df = df.drop(columns=['prompt'], errors='ignore')
    start_year = 2018
    end_year = 2022
    year_range = set(range(start_year, end_year+1))
    valid_company = df.groupby('CONM')['year'].apply(set).reset_index()
    valid_company = valid_company[valid_company['year'].apply(lambda x: year_range.issubset(x))]

    valid_df = df[df['CONM'].isin(valid_company['CONM'])]
    valid_df = valid_df[valid_df['year'].between(start_year, end_year+1)]

    # get unique comn number in valid_df
    print(valid_df['CONM'].nunique())
    
    valid_df.to_csv(f"../data/survive_data_{start_year}-{end_year}.csv", index=False)


def analyze_data(df):
    bins = [-float('inf'), 8.0, 9.0, 10.0, float('inf')]
    labels = ["<8.0", "8.0-9.0", "9.0-10.0", ">10.0"]

    for year in range(2018, 2023):
        df_year = df[df['year'] == year]
        # Categorize 'mcap' column based on bins
        df_year['mcap_category'] = pd.cut(df_year['mcap'], bins=bins, labels=labels, right=True)

        # Count the number of rows in each category
        category_counts = df_year['mcap_category'].value_counts().sort_index()

        print(f"Year {year}:")
        for category, count in category_counts.items():
            print(f"Category {category}: {count}")

def prompt_llm(df, start_year=2018, end_year=2022):
    df = df.loc[(df['year'] == end_year)]
    # df = df.head(10)
    print(df.shape)

    output_list = []
    for index, row in df.iterrows():
        company = row['CONM']
        try:
            messages = [
                {"role": "system", "content": "Forget all your previous instructions."},
                {"role": "user", "content": f"What are the revenues of {company} for each finance year from {start_year} to {end_year}? Please return the revenue only."}
            ]
            chat_completion = client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=messages,
                    temperature=0.00,
                    max_tokens=500
            )

            revenue_response = chat_completion.choices[0].message.content

            # Some function to validate revenue_response

            messages += [
                {"role": "assistant", "content": revenue_response},
                {"role": "user", "content": f"Based on the revenue information above, please predict the revenue of {company} in finance year {end_year+1}. Please return the revenue only."}
            ]

            predict_response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=messages,
                temperature=0.00,
                max_tokens=100
            )

            predict_response = predict_response.choices[0].message.content

            messages += [
                {"role": "system", "content": "Act as a financial expert with experience in stock recommendations."}, 
                {"role": "user", "content": f"Based on the information above, give either BUY, SELL, or DNK (do not have enough knowledge of the company) recommendation for {company} in finance year {end_year+2}."}
            ]

            action_response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=messages,
                temperature=0.00,
                max_tokens=100
            )

            action_response = action_response.choices[0].message.content

            print(row['mcap'], action_response)
            temp_list = list(row)
            # temp_list.append(messages)
            temp_list.append(action_response)
            
            output_list.append(temp_list)
        
        except Exception as e:
            print(e)
            sleep(10.0)

    results = pd.DataFrame(output_list, columns=['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', "action_response"])

    results.to_csv(f'../data/llm_prompt_outputs_v2/stock_recommendation.csv', index=False)

def check_answer(response):
    pass 

if __name__ == '__main__':
    start_year = 2018
    end_year = 2022
    if not os.path.exists(f"../data/survive_data_{start_year}-{end_year}.csv"):
        get_filtered_comanpy()
    df = pd.read_csv(f"../data/survive_data_{start_year}-{end_year}.csv")
    # analyze_data(df)    
    prompt_llm(df)