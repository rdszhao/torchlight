from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import yaml
import os

DIR = '../../data/full_pcaps'
os.makedirs(DIR, exist_ok=True)

url = 'https://mergetb.org/projects/searchlight/dataset/'
print('Acquiring links...')
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.pcap')]

with open('keywords.yaml', 'r') as f:
    file = yaml.safe_load(f)

keywords = file['keywords']
print(f"Filtering for links containing {keywords}")
filtered_links = [link for link in links if all(keyword in link for keyword in keywords)]

# 148 + 15 + ?
for link in tqdm(filtered_links[148 + 15 + 113 + 132:]):
    filename = link.split('/')[-1]
    filepath = os.path.join(DIR, filename)
    print(f'Downloading {filename}...')
    response = requests.get(link)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    print(f'{filename} downloaded to {filepath}')