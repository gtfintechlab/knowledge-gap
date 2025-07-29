import pandas as pd
from utils import *
import os
from together import Together
from time import sleep

def get_filtered_comanpy():
    model = 'chatgpt'
    df = pd.read_csv(f'../data/llm_prompt_outputs/{model}_yearly_combined.csv')
    df = df.drop(columns=['prompt_output'], errors='ignore')
    df = df.drop(columns=['prompt'], errors='ignore')
    start_year = 2010
    end_year = 2019
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

    for year in range(2010, 2020):
        df_year = df[df['year'] == year]
        # Categorize 'mcap' column based on bins
        df_year['mcap_category'] = pd.cut(df_year['mcap'], bins=bins, labels=labels, right=True)

        # Count the number of rows in each category
        category_counts = df_year['mcap_category'].value_counts().sort_index()

        print(f"Year {year}:")
        for category, count in category_counts.items():
            print(f"Category {category}: {count}")

def prompt_llm(df, start_year=2010, end_year=2019):
    client = Together()
    # get the unique company names
    company_names = df['CONM'].unique().tolist()
    
    for company in company_names:
        try:
            messages = [
                {"role": "system", "content": "Forget all your previous instructions."},
                {"role": "user", "content": f"What are the revenues of {company} for each finance year from {start_year} to {end_year}? Please return the revenue only."}
            ]
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=messages,
                temperature=0.01,
                max_tokens=1000
            )

            revenue_response = response.choices[0].message.content

            # Some function to validate revenue_response

            messages += [
                {"role": "assistant", "content": revenue_response},
                {"role": "user", "content": f"Based on the revenue information above, please predict the revenue of {company} in finance year {end_year+1}. Please return the revenue only."}
            ]

            predict_response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=messages,
                temperature=0.01,
                max_tokens=100
            )

            predict_response = predict_response.choices[0].message.content

            messages += [
                {"role": "system", "content": "You are a financial expert with stock recommendation experience."},
                {"role": "user", "content": f"Based on the information above, give either BUY, SELL, or DNK (do not have enough knowledge of the company) recommendation for {company} in finance year {end_year+1}."}
            ]

            action_response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=messages,
                temperature=0.01,
                max_tokens=100
            )

            action_response = action_response.choices[0].message.content
            print(action_response)
        
        except Exception as e:
            print(e)
            sleep(10.0)

def check_answer(response):
    pass 

if __name__ == '__main__':
    start_year = 2010
    end_year = 2019
    if not os.path.exists(f"../data/survive_data_{start_year}-{end_year}.csv"):
        get_filtered_comanpy()
    df = pd.read_csv(f"../data/survive_data_{start_year}-{end_year}.csv")
    # analyze_data(df)
    prompt_llm(df)