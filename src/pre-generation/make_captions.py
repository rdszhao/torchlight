import os
import json

data_dir = '../data'
img_dir = f"{data_dir}/finetune_imgs"
nprint_dir = f"{data_dir}/finetune_nprints"

captions = []
for file in os.listdir(nprint_dir):
	if '.nprint' in file:
		filename = file.split('-')[0] 
		new_name = f"{filename}.png"
		os.rename(f"{img_dir}/{file}", f"{img_dir}/{new_name}")
		caption = f"pixelated network data: {file.split('-')[-3]}, {file.split('.')[-4]}, {file.split('.')[-2]}" 
		entry = {'file_name': new_name, 'text': caption}
		captions.append(entry)

# write captions to csv using csv module
with open(f"{img_dir}/metadata.jsonl", 'w') as f:
	for entry in captions:
		json.dump(entry, f)
		f.write('\n')