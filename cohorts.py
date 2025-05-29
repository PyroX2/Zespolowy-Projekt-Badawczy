import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Acer\EMBED_dataset_tables\EMBED_OpenData_metadata.csv") #droga do csv

cohort2_ids = df.loc[df['cohort_num'] == 2, 'empi_anon'].unique()

n = len(cohort2_ids) // 2
np.random.seed(1)  
selected_ids = np.random.choice(cohort2_ids, size=n, replace=False)

mask = df['empi_anon'].isin(selected_ids) & (df['cohort_num'] == 2)
df.loc[mask, 'cohort_num'] = 3

df.to_csv('return_csv.csv', index=False)
