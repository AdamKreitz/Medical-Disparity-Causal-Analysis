import numpy as np
import requests
import urllib
import os
import logging
from pathlib import Path
import pandas as pd

file_name = 'all_cdc_data.csv'
url = 'https://chronicdata.cdc.gov/resource/g4ie-h725.csv'
file = requests.get(url)
open(f'src/data/{file_name}','wb').write(file.content)


