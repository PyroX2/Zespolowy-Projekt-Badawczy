import pandas as pd
import numpy as np

metadata_path = r"C:\Users\Acer\EMBED_dataset_tables\EMBED_OpenData_metadata.csv"

df = pd.read_csv(metadata_path)

best_seed = 982  
cohort2_ids = df.loc[df['cohort_num'] == 2, 'empi_anon'].unique()

np.random.seed(best_seed)
n = len(cohort2_ids) // 2
selected_to_3 = np.random.choice(cohort2_ids, size=n, replace=False)

patient_to_cohort = (
    df[['empi_anon', 'cohort_num']]
    .drop_duplicates(subset=['empi_anon'])
    .set_index('empi_anon')['cohort_num']
    .to_dict()
)

for pid in selected_to_3:
    patient_to_cohort[pid] = 3


df['cohort_num'] = df['empi_anon'].map(patient_to_cohort)

output_path = r"C:\Users\Acer\EMBED_dataset_tables\EMBED_metadata_seed{}_updated.csv".format(best_seed)
df.to_csv(output_path, index=False)

