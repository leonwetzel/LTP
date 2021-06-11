import numpy as np
import pandas as pd


data_file = "data/PSP_data.csv"

# load dataset
df = pd.read_csv(data_file, sep=',', quotechar='"')

df.loc[df['Category'] != "None", "Category"] = 'Offensive'
df.loc[df['Category'] == "None", "Category"] = 'Non-offensive'

df.to_csv("data/PSP_data.csv", index=False)