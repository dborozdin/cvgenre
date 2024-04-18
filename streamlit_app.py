import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import string
import time

data = [{'Geeks': 'dataframe', 'For': 'using', 'geeks': 'list'}, 
        {'Geeks':10, 'For': 20, 'geeks': 30}]  
  
df = pd.DataFrame.from_dict(data) 
df.head()
