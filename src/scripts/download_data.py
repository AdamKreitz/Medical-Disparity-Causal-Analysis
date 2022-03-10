import numpy as np
import urllib
import os
import logging
import regex as re
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils import SkeletonDiscovery
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.cit import fisherz
from causallearn.utils.cit import mv_fisherz
from causallearn.utils.cit import kci

#Load in csvs
def load_in_new_data(file_name):
    '''Load in data from file name and output as pandas dataframe'''
    cwd = os.getcwd()
    try:
        return pd.read_csv(f'{cwd}\src\data\{file_name}')
    except:
        return pd.read_excel(f'{cwd}\src\data\{file_name}')

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

def merge_year_data(prior_data, new_data):
    '''Function to merge prior and new data'''
    
    # Combine data sets together
    merged_df = pd.merge(prior_data.df,new_data.df,how = 'left',left_on = [prior_data.year_column,prior_data.country_column],right_on = [new_data.year_column, new_data.country_column])
    
    # Remove country and year column from new data if the column is not named the same as in prior data
    cols_to_remove = []
    if new_data.year_column != prior_data.year_column:
        cols_to_remove.append(new_data.year_column)
    if new_data.country_column != prior_data.country_column:
        cols_to_remove.append(new_data.country_column)
    if len(cols_to_remove) > 0:
        merged_df = merged_df.drop(columns = cols_to_remove)
    

    # Clean merged data to only have countries in old and new data
    old_countries = list(prior_data.df[prior_data.country_column].unique())
    new_countries = list(new_data.df[new_data.country_column].unique())

    def check_country(x):
        '''Helper function to make sure countries in data were in both datasets merged together'''
        if x in old_countries and x in new_countries:
            return True
        else:
            return False

    merged_df = merged_df[list(map(check_country,list(merged_df[prior_data.country_column])))]

    merged_dataset = dataset(merged_df)

    new_min_year = max(prior_data.min_year,new_data.min_year)
    new_max_year = min(prior_data.max_year, new_data.max_year)

    if new_max_year < new_min_year:
        print('Datasets not compatible since max year of one is less than min year of another')
    else:
        merged_dataset.set_min(new_min_year)
        merged_dataset.set_max(new_max_year)

    return merged_dataset

def merge_no_year_data(prior_data, new_data):
    '''Function to merge prior and new data'''
    
    # Combine data sets together
    merged_df = pd.merge(prior_data.df,new_data.df,how = 'left',left_on = [prior_data.country_column],right_on = [new_data.country_column])
    
    # Remove country and year column from new data if the column is not named the same as in prior data
    cols_to_remove = []
    if new_data.country_column != prior_data.country_column:
        cols_to_remove.append(new_data.country_column)
    if len(cols_to_remove) > 0:
        merged_df = merged_df.drop(columns = cols_to_remove)
    

    # Clean merged data to only have countries in old and new data
    old_countries = list(prior_data.df[prior_data.country_column].unique())
    new_countries = list(new_data.df[new_data.country_column].unique())

    def check_country(x):
        '''Helper function to make sure countries in data were in both datasets merged together'''
        if x in old_countries and x in new_countries:
            return True
        else:
            return False

    merged_df = merged_df[list(map(check_country,list(merged_df[prior_data.country_column])))]

    merged_dataset = dataset(merged_df)

    return merged_dataset

def reformat_data(dataset):
    '''Reformat dataframe to be indexed by country with features labeled by feature and year'''

    years_present_df = dataset.df[(dataset.df[dataset.year_column] >= dataset.min_year) & (dataset.df[dataset.year_column] <= dataset.max_year)].groupby(dataset.country_column).count()[[dataset.year_column]]
    
    # Only include countries with at least 10 years in data (We made this cutoff but it could be altered especially if less years are in the dataset)
    countries_to_include = list(years_present_df[years_present_df[dataset.year_column] > 10].index)

    included_years = [i for i in range(int(dataset.min_year),int(dataset.max_year) + 1)]
    columns = dataset.df.columns.drop([dataset.country_column, dataset.year_column,'index'])
    new_col_names = []
    for col in columns:
        for i in included_years:
            year = str(i)
            col_name = f'{col}_{year}'
            new_col_names.append(col_name)

    data = {}
    countries_to_remove = []
    groups = dataset.df.groupby(dataset.country_column)
    for key, group in groups:
        if key in countries_to_include:
            row_data = []
            c_group = group.drop(columns = dataset.country_column)
            c_group = c_group.set_index(dataset.year_column)
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
                    X = pd.DataFrame({dataset.year_column:col_years})
                    Y = pd.DataFrame({col:col_vals})[col]
                    col_predictor = LinearRegression().fit(X,Y)
                    for year in included_years:
                        try:
                            t = c_group.loc[year,col]
                            if t == t:
                                row_data.append(t)
                            else:
                                predicted_value = col_predictor.predict(pd.DataFrame({dataset.year_column:[year]}))[0]    
                                row_data.append(predicted_value) 
                        except:
                            predicted_value = col_predictor.predict(pd.DataFrame({dataset.year_column:[year]}))[0]    
                            row_data.append(predicted_value) 
                else:
                    for year in included_years:
                        row_data.append(np.nan) 
                    countries_to_remove.append(key)
                    print(f'{key} has no data for {col} so it was removed from the data.')
            data[key] = row_data
        

    reformatted_df = pd.DataFrame.from_dict(data, orient='index',columns = new_col_names)
    reformatted_df.reset_index(inplace=True)
    reformatted_df = reformatted_df.rename(columns = {'index':dataset.country_column})

    # Remove any country that is missing all data for any of the included feature
    def check_country(x):
        '''Helper function to make sure countries in data were in both datasets merged together'''
        if x in countries_to_remove:
            return False
        else:
            return True

    reformatted_df = reformatted_df[list(map(check_country,list(reformatted_df.index)))]

    ttl_nulls = pd.DataFrame(reformatted_df.isnull().sum()).sum()
    ttl_vals = reformatted_df.shape[0] * reformatted_df.shape[1]
    pct_missing = str(ttl_nulls / ttl_vals)
    print(f'Percentage missing = {pct_missing}')
    return reformatted_df

def find_name(pattern, columns):
        '''Helper function to find a pattern in colum names to be 
        able to find country name column and year column'''
        for col in columns:
            t = re.findall(pattern, col)
            if len(t) > 0:
                return col
        return np.nan

class dataset():
    def __init__(self,df):
        self.df = df.reset_index()
        # Find column names for country and year for dataset
        country_pattern = '[Cc]ountry' 
        year_pattern = '[Yy]ear'  
        self.country_column = find_name(country_pattern, self.df.columns)
        self.year_column = find_name(year_pattern, self.df.columns)

        # Clean up data to make all columns have type float (If not numerical columns they will be removed later)
        self.df = self.df.applymap(convert_to_float)

        # Clean up country names that have common errors, but may not catch all errors in naming conventions
        self.df[self.country_column] = self.df[[self.country_column]].applymap(fix_name)
        
        # Get rid of any non-numerical columns beside country name column
        try:
            self.df = self.df.drop(columns = ['Unnamed: 0'])
        except:
            self.df = self.df
        try:
            self.df = self.df.drop(columns = ['index_x'])
        except:
            self.df = self.df
        columns_to_drop = []
        for col in self.df.columns:
            if type(self.df[col][0]) != np.float64:
                columns_to_drop.append(col)
            elif self.df[col].nunique() <= 2:
                columns_to_drop.append(col)
           
        columns_to_drop.remove(self.country_column)
        if len(columns_to_drop) > 0:
            self.df = self.df.drop(columns = columns_to_drop)

        if self.year_column == self.year_column:
            # Get rid of entities with null values for country or year
            self.df.dropna(subset = [self.country_column, self.year_column])
            # Set min and max years included in the dataset
            self.min_year = self.df[self.year_column].min()
            self.max_year = self.df[self.year_column].max()
        else:
            # Get rid of entities with null values for country column since no year column
            self.df.dropna(subset = [self.country_column])
            # Set min and max years included in the dataset
            self.min_year = self.year_column
            self.max_year = self.year_column
           
    def set_min(self,new_min_year):
        '''Internal function to reset min year of data'''
        self.min_year = new_min_year
    
    def set_max(self,new_max_year):
        '''Internal function to reset min year of data'''
        self.max_year = new_max_year


def create_new_data(file_names,output_file_name):
    '''Creates a cleaned and properly formatted version of Wall inputted file names'''
    # Seperate data sets into those that have a year column and those that do not
    datasets_with_years = []
    datasets_without_years = []
    for file_name in file_names:
        df = load_in_new_data(file_name)
        new_dataset = dataset(df)
        if new_dataset.year_column == new_dataset.year_column:
            datasets_with_years.append(new_dataset)
        else:
            datasets_without_years.append(new_dataset)
    
    # Merge all datasets with years
    while len(datasets_with_years) > 1:
        merged_dataset = merge_year_data(datasets_with_years[0],datasets_with_years[1])
        datasets_with_years = [merged_dataset] + datasets_with_years[2:]

    # Reformat data with years
    if len(datasets_with_years) == 1:
        reformatted_data = dataset(reformat_data(datasets_with_years[0]))
        datasets_without_years = [reformatted_data] + datasets_without_years
    
    # Merge all data
    while len(datasets_without_years) > 1:
        merged_dataset = merge_no_year_data(datasets_without_years[0],datasets_without_years[1])
        datasets_without_years = [merged_dataset] + datasets_without_years[2:]
    final_dataset = datasets_without_years[0]
    try:
        final_dataset.df = final_dataset.df.drop(columns = ['index'])
    except:
        final_dataset.df = final_dataset.df
    cwd = os.getcwd()
    output_file_path = f'{cwd}/src/final_data/{output_file_name}.csv'
    final_dataset.df.to_csv(output_file_path)
    print(f'Successfully made new csv file of final data called {output_file_name} in final_data folder.')
    return final_dataset

# Define function to Run PC (Default is fisherz test with no missing values but can be changed)
def run_pc_on_data(input_dataset, file_name, alpha=0.05, func_type = fisherz):
    input_dataset.df = input_dataset.df.set_index(input_dataset.country_column)

    # Run initial PC to get nodes
    cg = pc(np.array(input_dataset.df), alpha, func_type, True, uc_rule=1)
    
    tier_list = {}
    for col in input_dataset.df.columns:
        try:
            col_year = int(col[-4:]) - input_dataset.min_year
        except:
            col_year = 0
        tier_list[col] = col_year

    ## Rename the nodes

    nodes = cg.G.get_nodes()
    names = list(input_dataset.df.columns)
    for i in range(len(nodes)):

        node = nodes[i]
        name = names[i]
        node.set_name(name)

    ## Create the tiers

    nodes = cg.G.get_nodes()
    names = list(input_dataset.df.columns)
  
    bk = BackgroundKnowledge()
    for i in range(len(nodes)):
        node = nodes[i]
        name = names[i] 

        t = tier_list[name]
        bk = bk.add_node_to_tier(node,int(t))
    cg = pc(np.array(input_dataset.df), alpha, func_type, True, mvpc=False, uc_rule=1, background_knowledge = bk)
    cg.to_nx_graph()
    cg.draw_nx_graph(skel=False)

    connection_dict = {}
    def add_to_dict(dict, key):
        if key in dict.keys():
            dict[key] += 1
        else:
            dict[key] = 1

    for connection in cg.find_fully_directed():
        # connection is tuple with first value as index of first column in connection and second value as second column in connection
        first_col = input_dataset.df.columns[int(connection[0])]
        second_col = input_dataset.df.columns[int(connection[1])]
        try:
            first_col_year = int(first_col[-4:]) - input_dataset.min_year
            first_col_feature = first_col[:-5]
        except:
            first_col_year = np.nan
            first_col_feature = first_col
        try:
            second_col_year = int(second_col[-4:]) - input_dataset.min_year
            second_col_feature = second_col[:-5]
        except:
            second_col_year = np.nan
            second_col_feature = second_col
        
        if (first_col_year != first_col_year) or (second_col_year != second_col_year):
            year_diff = -1
            relation = tuple([first_col_feature, second_col_feature, year_diff])
        elif first_col_year > second_col_year:
            year_diff = first_col_year - second_col_year
            relation = tuple([second_col_feature, first_col_feature, year_diff])
        else:
            year_diff = second_col_year - first_col_year
            relation = tuple([first_col_feature, second_col_feature, year_diff])

        add_to_dict(connection_dict, relation)
    
    relation_df = pd.DataFrame.from_dict(connection_dict,orient = 'index',columns = ['Frequency']).reset_index()
    relation_df['Feature 1'] = relation_df[['index']].applymap(lambda x : x[0])
    relation_df['Feature 2'] = relation_df[['index']].applymap(lambda x : x[1])
    relation_df['Year Difference'] = relation_df[['index']].applymap(lambda x : x[2])
    relation_df = relation_df.drop(columns='index')
    relation_df = relation_df.sort_values(by = 'Frequency', ascending = False)
    cwd = os.getcwd()
    output_file_path = f'{cwd}/src/Results/{file_name}.csv'
    relation_df.df.to_csv(output_file_path)
    print(f'Successfully made new csv file of results called {file_name} in Results folder.')
    return relation_df

# Add your own data here
new_file_names = ['your_file_name']
new_run_name = 'your_name_for_output_file.csv'
#final_dataset = create_new_data(file_names,new_run_name)
#relations = run_pc_on_data(final_dataset, run_name, alpha = 0.2)
#print(relations)
