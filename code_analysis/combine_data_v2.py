import re
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm

from utils import *

combine_files(['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', 'prompt', 'prompt_output'], 
              "../data/llm_prompt_outputs_v2/llama8b_yearly_3/", 
              '../data/llm_prompt_outputs_v2/llama8b_yearly_3_combined.csv')

combine_files(['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', 'prompt', 'prompt_output'], 
              "../data/llm_prompt_outputs_v2/gpt4o_mini_yearly/", 
              '../data/llm_prompt_outputs_v2/gpt4o_mini_yearly_combined.csv')

combine_files(['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', 'prompt', 'prompt_output'], 
              "../data/llm_prompt_outputs_v2/llama70b_yearly_3/", 
              '../data/llm_prompt_outputs_v2/llama70b_yearly_3_combined.csv')

combine_files(['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', 'prompt', 'prompt_output'], 
              "../data/llm_prompt_outputs_v2/gpt4o_yearly/", 
              '../data/llm_prompt_outputs_v2/gpt4o_yearly_combined.csv')

combine_files(['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', 'prompt', 'prompt_output'], 
              "../data/llm_prompt_outputs_v2/gpt45_yearly/", 
              '../data/llm_prompt_outputs_v2/gpt45_yearly_combined_small.csv')

combine_files(['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', 'prompt', 'prompt_output'], 
              "../data/llm_prompt_outputs_v2/deepseek_v3_yearly/", 
              '../data/llm_prompt_outputs_v2/deepseek_v3_yearly_combined_small.csv')

combine_files(['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', 'prompt', 'prompt_output'], 
              "../data/llm_prompt_outputs_v2/gemini15_yearly_small/", 
              '../data/llm_prompt_outputs_v2/gemini15_yearly_combined_small.csv')

combine_files(['CONM', 'year', 'mcap', 'revt', 'PERMNO', 'GVKEY', 'TIC', 'SIC', 'CIK', 'prompt', 'prompt_output'], 
              "../data/llm_prompt_outputs_v2/finma_7b_yearly/", 
              '../data/llm_prompt_outputs_v2/finma_7b_yearly_combined_small.csv')