import numpy as np
import requests
import urllib
import os
import logging
from pathlib import Path
import pandas as pd

cwd = os.getcwd()
file_name = 'all_cdc_data.csv'
df = pd.read_csv(f'{cwd}\src\data\{file_name}')
mn = df['yearstart'].min()
mx = df['yearend'].max()
print(mx)