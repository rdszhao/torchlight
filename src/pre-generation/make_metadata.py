import os
import json

data_dir = '../../data'
img_dir = f"{data_dir}/finetune_imgs"

captions = []
for file in os.listdir(img_dir):
	if '.png' in file:
		# filename = file.split('-')[0] 
		# new_name = f"{filename}.png"
		# os.rename(f"{img_dir}/{file}", f"{img_dir}/{new_name}")
		caption = f"pixelated network data: {file.split('-')[-3]}, {file.split('.')[-4]}, {file.split('.')[-3]}, {file.split('.')[-2]}" 
		entry = {'file_name': file, 'text': caption}
		captions.append(entry)

with open(f"{img_dir}/metadata.jsonl", 'w') as f:
	for entry in captions:
		json.dump(entry, f)
		f.write('\n')