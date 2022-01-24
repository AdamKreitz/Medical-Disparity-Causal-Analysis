import numpy as np
import requests
import urllib
import os
import logging
from pathlib import Path
import pandas as pd

# Load in csvs
cwd = os.getcwd()
WHR_file_name = 'world-happiness-report.csv'
WHR2021_file_name = 'world-happiness-report-2021.csv'
CPDS_file_name = 'CPDS_1960-2019_Update_2021.xlsx'
WHR_df = pd.read_csv(f'{cwd}\src\data\{WHR_file_name}')
WHR2021_df = pd.read_csv(f'{cwd}\src\data\{WHR2021_file_name}')
CPDS_df = pd.read_excel(f'{cwd}\src\data\{CPDS_file_name}')


# Clean Data
WHR_df = WHR_df.drop(columns = ['Positive affect','Negative affect'])
WHR2021_df['year'] = 2021
WHR2021_df = WHR2021_df.drop(columns = ['Standard error of ladder score','upperwhisker','lowerwhisker','Ladder score in Dystopia',
                          'Explained by: Log GDP per capita','Explained by: Social support','Explained by: Healthy life expectancy',
                          'Explained by: Freedom to make life choices','Explained by: Generosity','Explained by: Perceptions of corruption',
                          'Dystopia + residual'])
WHR2021_df = WHR2021_df.rename(columns = {'Ladder score':'Life Ladder','Logged GDP per capita':'Log GDP per capita',
                            'Healthy life expectancy':'Healthy life expectancy at birth'})

dic = {}
for i in WHR2021_df['Regional indicator'].unique():
    dic[i]=list(WHR2021_df[WHR2021_df['Regional indicator'] ==i].groupby('Country name').size().index)
dic['Sub-Saharan Africa'].append('Angola')
dic['Latin America and Caribbean'].append('Belize')
dic['South Asia'].append('Bhutan')
dic['Sub-Saharan Africa'].append('Central African Republic')
dic['Sub-Saharan Africa'].append('Congo (Kinshasa)')
dic['Latin America and Caribbean'].append('Cuba')
dic['Sub-Saharan Africa'].append('Djibouti')
dic['Latin America and Caribbean'].append('Guyana')
dic['Middle East and North Africa'].append('Oman')
dic['Middle East and North Africa'].append('Qatar')
dic['Sub-Saharan Africa'].append('Somalia')
dic['Sub-Saharan Africa'].append('Somaliland region')
dic['Sub-Saharan Africa'].append('South Sudan')
dic['Middle East and North Africa'].append('Sudan')
dic['Latin America and Caribbean'].append('Suriname')
dic['Middle East and North Africa'].append('Syria')
dic['Latin America and Caribbean'].append('Trinidad and Tobago')

def find_region(x):
    '''Helper Function to Return Region Name based on the Country Name entered'''
    for reg in dic.keys():
        for c in dic[reg]:
            if x == c:
                return reg
WHR_df['Regional indicator'] = WHR_df['Country name'].apply(find_region)

cmbd_WHR_df = pd.concat([WHR_df,WHR2021_df])

CPDS_columns_to_drop = ['year_01','country_01','elect','year_02','country_02']
for i in range(1,11):
    num = str(i)
    CPDS_columns_to_drop.append(f'sconserv{num}')
CPDS_df = CPDS_df.drop(columns = CPDS_columns_to_drop)

print(CPDS_df.head())