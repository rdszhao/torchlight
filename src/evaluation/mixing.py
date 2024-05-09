# %%
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
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
# %%
data_dir = '../../data'
real_nprints_dir = f"{data_dir}/finetune_nprints"
synth_nprints_dir = f"{data_dir}/generated_nprint"
mix_output_dir = 'results/mix_results.json'
synth_output_dir = 'results/synth_results.json'
realfake_output_dir = 'results/realfake_results.json'

realdata, realkeys = get_datafiles(real_nprints_dir, return_keys=True)
synthdata, synthkeys = get_datafiles(synth_nprints_dir, return_keys=True)

def mix_data(realdata, synthdata, realkeys, synthkeys, n=32):
	n = int(n * realdata['data'].shape[0])
	synarr = np.array([f"{item.split('_')[0]}.nprint" for item in synthkeys])
	to_replace = np.random.choice(len(realkeys), n, replace=False)
	available_indices = [(synarr == realkeys[idx]).nonzero()[0] for idx in to_replace]
	replacing_indices = [np.random.choice(indices, 1)[0] for indices in available_indices]

	copydata = realdata['data'].copy()
	copydata[to_replace] = synthdata['data'][replacing_indices]
	return copydata
# %%
mixing_rates = np.arange(0, 1, 0.1)
results = defaultdict(list)
for label, Y in tqdm(realdata['labels'].items()):
	print(f"evaluating models on {label}...\n")
	for rate in mixing_rates:
		indices = np.arange(realdata['data'].shape[0])
		X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(realdata['data'], Y, indices, test_size=0.2, random_state=42)
		X_train = mix_data({'data': X_train}, synthdata, np.array(realkeys)[idx_train], synthkeys, n=rate)
		model = RandomForestClassifier(n_estimators=100, random_state=42)
		print(f"training on {label} with a mixing rate of {rate}...")
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		results[label].append((rate, acc))

with open(mix_output_dir, 'w') as f:
	json.dump(results, f, indent=4)

print('trained on mixed data, tested on real data')
print(json.dumps(results, indent=4))
# %%
fig, ax = plt.subplots()
for key, entry in results.items():
	sns.lineplot(x=[item[0] for item in entry], y=[item[1] for item in entry], label=key, ax=ax, marker='o')

ax.legend(title='label', loc='upper right', bbox_to_anchor=(1.3, 1.02))
fig.suptitle('RF Trained on Mixed Data, Tested on Real Data')
plt.xlabel('Mixing Rate')
plt.xticks(np.arange(0, 1, 0.1))
plt.ylabel('Accuracy')
plt.show()
# %%
for label, values in realdata['labels'].items():
	uniques, counts = np.unique(values, return_counts=True)
	print(f"{label}: {list(counts)}")
	# print(f"{label} balance deviations: {list((1 / len(uniques)) - (counts / len(values)))}")