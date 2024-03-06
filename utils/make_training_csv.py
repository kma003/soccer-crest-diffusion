import os
import pandas as pd


data_dir = os.path.join(os.path.dirname(__file__),'..','data')
crest_dir = os.path.join(data_dir,'crests')

# Get all visible directories within the crest directory
visible_directories = []
for item in os.listdir(crest_dir):
    if os.path.isdir(os.path.join(crest_dir,item)) and not item.startswith('.'):
        visible_directories.append(item)

# Fill out the training csv
df = pd.DataFrame(columns=['Country','Team','Image Path'],dtype=str)
for country_name in visible_directories:
    full_path = os.path.normpath(os.path.join(crest_dir,country_name))
    for item in os.listdir(full_path):
        item_path = os.path.join(full_path,item)
        if os.path.isfile(item_path) and item.lower().endswith('.png'):
            team_name = os.path.splitext(item)[0]
            new_row = {'Country':country_name,'Team':team_name,'Image Path':item_path}
            df = df.append(new_row,ignore_index=True)

# Save as csv
df.to_csv(os.path.join(data_dir,'training_data.csv'),index=False)
