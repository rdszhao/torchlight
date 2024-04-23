import numpy as np
import json
from collections import defaultdict
from clf_tools import get_datafiles
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dir = '../../data'
real_nprints_dir = f"{data_dir}/finetune_nprints"
synth_nprints_dir = f"{data_dir}/replayable_generated_nprints"
base_output_dir = 'results/base_results.json'
synth_output_dir = 'results/synth_results.json'
realfake_output_dir = 'results/realfake_results.json'
datafiles = get_datafiles(real_nprints_dir)
traindata = get_datafiles(synth_nprints_dir, synth=True)

results = defaultdict(dict)
for label, Y in tqdm(datafiles['labels'].items()):
	X_train, X_test, y_train, y_test = train_test_split(datafiles['data'], Y, test_size=0.2, random_state=42)
	models = {
		'rf': RandomForestClassifier(n_estimators=100, random_state=42),
		'svc': SVC(kernel='linear', C=1.0, random_state=42),
		'xgb': XGBClassifier(objective='multi:softmax', num_class=len(np.unique(Y)), use_label_encoder=False, seed=42)
	}
	for name, model in tqdm(models.items()):
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		results[label][name] = acc

with open(base_output_dir, 'w') as f:
	json.dump(results, f, indent=4)

print('trained on real data, tested on real data')
print(json.dumps(results, indent=4))

testdata = datafiles
results = defaultdict(dict)
for label, (y_train, y_test) in tqdm(zip(traindata['labels'].keys(), zip(traindata['labels'].values(), testdata['labels'].values()))):
	X_train, X_test = traindata['data'], testdata['data']
	models = {
		'rf': RandomForestClassifier(n_estimators=100, random_state=42),
		'svc': SVC(kernel='linear', C=1.0, random_state=42),
		'xgb': XGBClassifier(objective='multi:softmax', num_class=len(np.unique(Y)), use_label_encoder=False, seed=42)
	}
	for name, model in tqdm(models.items()):
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		results[label][name] = acc

with open(synth_output_dir, 'w') as f:
	json.dump(results, f, indent=4)

print('trained on synthetic data, tested on real data')
print(json.dumps(results, indent=4))

testdata = datafiles
results = defaultdict(dict)
for label, (y_train, y_test) in tqdm(zip(traindata['labels'].keys(), zip(traindata['labels'].values(), testdata['labels'].values()))):
	X_train, X_test = traindata['data'], testdata['data']
	models = {
		'rf': RandomForestClassifier(n_estimators=100, random_state=42),
		'svc': SVC(kernel='linear', C=1.0, random_state=42),
		'xgb': XGBClassifier(objective='multi:softmax', num_class=len(np.unique(Y)), use_label_encoder=False, seed=42)
	}
	for name, model in tqdm(models.items()):
		model.fit(X_test, y_test)
		y_pred = model.predict(X_train)
		acc = accuracy_score(y_train, y_pred)
		results[label][name] = acc

with open(realfake_output_dir, 'w') as f:
	json.dump(results, f, indent=4)

print('trained on real data, tested on synthetic data')
print(json.dumps(results, indent=4))