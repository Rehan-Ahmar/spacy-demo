import pandas
import os

df = pandas.read_excel(r'C:\Users\Rehan.Ahmar\Desktop\Ner training Data.xlsx')
print(df.shape)

mask = df['sentence'].str.endswith('(...)"') | df['trained_entity_data'].str.endswith('(...)"') \
     | df['sentence'].str.endswith('(...)') | df['trained_entity_data'].str.endswith('(...)')
incomplete_rows = df[mask]
complete_rows = df[~mask]
incomplete_rows.to_excel(r'C:\Users\Rehan.Ahmar\Desktop\Incomplete data.xlsx')
complete_rows.to_excel(r'C:\Users\Rehan.Ahmar\Desktop\Complete data.xlsx')

train_data_dict = {int(k): v for k, v in complete_rows.groupby('clause_id')}
for key, value in train_data_dict.items():
    filename = 'clause_id_' + str(key) + '.xlsx' 
    value.to_excel(os.path.join(r'C:\Users\Rehan.Ahmar\Desktop', filename))
