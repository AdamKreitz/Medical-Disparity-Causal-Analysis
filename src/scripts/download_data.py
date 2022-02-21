import numpy as np
import urllib
import os
import logging
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression

#Load in csvs
def load_in_new_data(file_name):
    '''Load in data from file name and output as pandas dataframe'''
    cwd = os.getcwd()
    try:
        return pd.read_csv(f'{cwd}\src\data\{file_name}')
    except:
        return pd.read_excel(f'{cwd}\src\data\{file_name}')

WHR_file_name = 'world-happiness-report.csv'
WHR2021_file_name = 'world-happiness-report-2021.csv'
WHR_df = load_in_new_data(WHR_file_name)
WHR2021_df = load_in_new_data(WHR2021_file_name)

# Clean WHR Data
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

# Merge WHR data with new data
def merge_data(prior_data, new_data, new_year_column_name, new_country_column_name):
    '''Function to merge prior and new data'''

    def convert_to_float(x):
        '''Converts inputted value to float if possible'''
        try:
            return float(x)
        except:
            return x

    def fix_name(x):
        '''fix country names to be in same format as WHR'''
        if x == 'USA':
            return 'United States'
        else:
            return x
    
    new_data = new_data.dropna(subset = [new_year_column_name, new_country_column_name])
    new_data[new_year_column_name] = new_data[[new_year_column_name]].applymap(convert_to_float)
    new_data[new_country_column_name] = new_data[[new_country_column_name]].applymap(fix_name)

    merged_df = pd.merge(prior_data,new_data,how = 'left',left_on = ['year','Country name'],right_on = [new_year_column_name, new_country_column_name])
    cols_to_remove = []
    if new_year_column_name != 'year':
        cols_to_remove.append(new_year_column_name)
    if new_country_column_name != 'Country name':
        cols_to_remove.append(new_country_column_name)
    if len(cols_to_remove) > 0:
        merged_df = merged_df.drop(columns = cols_to_remove)
    cleaned_merged_df = merged_df.applymap(convert_to_float)
    return cleaned_merged_df

def reformat_data(df):
    temp_df = df.drop(columns = ['Regional indicator'])
    years_present_df = temp_df[(temp_df['year'] >= 2007) & (temp_df['year'] <= 2021)].groupby('Country name').count()[['year']]
    countries_to_include = list(years_present_df[years_present_df['year'] > 10].index)
    included_years = [i for i in range(2007,2022)]
    columns = temp_df.columns[2:]
    new_col_names = []
    for col in columns:
        for i in included_years:
            year = str(i)
            col_name = f'{col}_{year}'
            new_col_names.append(col_name)

    groups = temp_df.groupby('Country name')

    num_cols = len(columns)
    data = {}
    for key, group in groups:
        if key in countries_to_include:
            row_data = []
            c_group = group.drop(columns = 'Country name')
            c_group = c_group.set_index('year')
            for col in columns:
                col_years = []
                col_vals = []
                for year in included_years:
                    try:
                        t = c_group.loc[year,col]
                        if t == t:
                            col_years.append(year)
                            col_vals.append(t)
                    except:
                        t = 0  
                if len(col_years) > 0:
                    X = pd.DataFrame({'year':col_years})
                    Y = pd.DataFrame({col:col_vals})[col]
                    col_predictor = LinearRegression().fit(X,Y)
                    for year in included_years:
                        try:
                            t = c_group.loc[year,col]
                            if t == t:
                                row_data.append(t)
                            else:
                                predicted_value = col_predictor.predict(pd.DataFrame({'year':[year]}))[0]    
                                row_data.append(predicted_value) 
                        except:
                            predicted_value = col_predictor.predict(pd.DataFrame({'year':[year]}))[0]    
                            row_data.append(predicted_value) 
                else:
                    for year in included_years:
                        row_data.append(np.nan) 
            data[key] = row_data
        

    reformatted_data = pd.DataFrame.from_dict(data, orient='index',columns = new_col_names)

    ttl_nulls = pd.DataFrame(reformatted_data.isnull().sum()).sum()
    ttl_vals = reformatted_data.shape[0] * reformatted_data.shape[1]
    pct_missing = str(ttl_nulls / ttl_vals)
    print(f'Percentage missing = {pct_missing}')
    return reformatted_data

def clean_reformatted_data(reformatted_df,new_df,new_country_column_name):
    '''Clean reformatted data to only have countries in old and new data'''
    old_countries = list(cmbd_WHR_df.sort_values(by='Country name')['Country name'].unique())
    new_countries = list(new_df.sort_values(by=new_country_column_name)[new_country_column_name].unique())

    def check_country(x):
        if x in old_countries and x in new_countries:
            return True
        else:
            return False
    cleaned_refformatted_df = reformatted_df[list(map(check_country,list(reformatted_df.index)))]
    ttl_nulls = pd.DataFrame(cleaned_refformatted_df.isnull().sum()).sum()
    ttl_vals = cleaned_refformatted_df.shape[0] * cleaned_refformatted_df.shape[1]
    pct_missing = str(ttl_nulls / ttl_vals)
    print(f'Percentage missing = {pct_missing}')
    return cleaned_refformatted_df

def create_new_data(file_name,new_year_column_name, new_country_column_name,output_file_name):
    '''Creates a cleaned and properly formatted version of WHR data combined with new data'''
    new_df = load_in_new_data(file_name)
    all_data = merge_data(cmbd_WHR_df,new_df,new_year_column_name, new_country_column_name)
    reformatted_data = reformat_data(all_data)
    cleaned_reformatted_data = clean_reformatted_data(reformatted_data,new_df,'year')
    cwd = os.getcwd()
    output_file_path = f'{cwd}/src/final_data/{output_file_name}'
    cleaned_reformatted_data.to_csv(output_file_path)
    print(f'Successfully made new csv file from {file_name} called {output_file_name}')

# Make final csv with all WHR data
cwd = os.getcwd()
WHR_data = reformat_data(cmbd_WHR_df)
final_WHR_path = f'{cwd}/src/final_data/WHR_with_no_missing_data.csv'
WHR_data.to_csv(final_WHR_path)
print(f'Successfully made new csv file for WHR data called WHR_with_no_missing_data.csv')

# Make final csv with all WHR and CPDS data
CPDS_file_name = 'cleaned_CPDS.xlsx'
create_new_data(CPDS_file_name,'year','country','gov_data_with_no_missing_data.csv')



