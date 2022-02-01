import numpy as np
import urllib
import os
import logging
from pathlib import Path
import pandas as pd
from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils import SkeletonDiscovery
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.cit import fisherz
from causallearn.utils.cit import mv_fisherz
from causallearn.utils.cit import kci

WHR_file_name = 'world-happiness-report.csv'
WHR2021_file_name = 'world-happiness-report-2021.csv'
CPDS_file_name = 'CPDS_1960-2019_Update_2021.xlsx'
WHR_df = pd.read_csv('final_data/Cleaned_WHD.csv')
#WHR2021_df = pd.read_csv('data\{WHR2021_file_name}')
#CPDS_df = pd.read_excel('data\{CPDS_file_name}')


# Clean Data
#WHR_df = WHR_df.drop(columns = ['Positive affect','Negative affect'])
#WHR2021_df['year'] = 2021
#WHR2021_df = WHR2021_df.drop(columns = ['Standard error of ladder score','upperwhisker','lowerwhisker','Ladder score in Dystopia',
 #                         'Explained by: Log GDP per capita','Explained by: Social support','Explained by: Healthy life expectancy',
  #                        'Explained by: Freedom to make life choices','Explained by: Generosity','Explained by: Perceptions of corruption',
  #                        'Dystopia + residual'])
#WHR2021_df = WHR2021_df.rename(columns = {'Ladder score':'Life Ladder','Logged GDP per capita':'Log GDP per capita',
                #            'Healthy life expectancy':'Healthy life expectancy at birth'})

WHR_df = WHR_df.rename(columns = {'Unnamed: 0': 'Country'})
WHR_df = WHR_df.set_index('Country')

cols = []
labs = ['2006', '2009', '2012', '2015', '2018', '2021']
for i in WHR_df.columns:
    if i[-4:] in labs:
        cols.append(i)
test = WHR_df[cols]

for i in test.columns:
    test[i].apply(float)

test = test.drop(['Healthy life expectancy at birth_2015'], axis=1)

cg = pc(np.array(test), 0.05, \
   mv_fisherz, True, mvpc=True, uc_rule=1)

tier_list = {}
for i in WHR_df.columns:
    tier_list[i] = (int(i[-4:]) - 2005)
    
## Rename the nodes
    
nodes = cg.G.get_nodes()
names = list(WHR_df.columns)
for i in range(len(nodes)):
    
    node = nodes[i]
    name = names[i]
    node.set_name(name)
    
## Create the tiers

nodes = cg.G.get_nodes()
names = list(WHR_df.columns)
tier = {}
bk = BackgroundKnowledge()
for i in range(len(nodes)):
   #print(i.get_name())
    #print(d[int(i.get_name()[1])-1][1])
    
    node = nodes[i]
    name = names[i]
    #node.set_name(name)
    #tier[i.get_name()] = int(((d[int(i.get_name()[1:])-1][1])/3)+9) 
    
    t = tier_list[name]
    bk = bk.add_node_to_tier(node,int(t))

cg = pc(np.array(test), 0.15, \
   mv_fisherz, True, mvpc=True, uc_rule=0, background_knowledge = bk)
cg.to_nx_graph()
cg.draw_nx_graph(skel=False)