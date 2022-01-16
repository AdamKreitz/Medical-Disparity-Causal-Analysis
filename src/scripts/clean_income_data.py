import numpy as np
import urllib
import os
import logging
from pathlib import Path
import pandas as pd

cwd = os.getcwd()
file_name = 'income_data.csv'
df = pd.read_csv(f'{cwd}\src\data\{file_name}')
df = df.drop(columns = ['Unnamed: 11'])
df = df.dropna(axis = 0, thresh = 2)
df = df.set_index('State')

def clean_values(x):
    try:
        x =x.replace('$','')
    except:
        print(f'Unexpected error cleaning {x}')
    try:
        x = x.replace(',','')
    except:
        print(f'Unexpected error cleaning {x}') 
    try:
        x = int(x)  
    except:
        print(f'Unexpected error converting {x} to int') 
    return x

df = df.applymap(clean_values)

print(df.head())