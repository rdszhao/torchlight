# %%
from bs4 import BeautifulSoup
import requests
import os

DIR = '../data/full_pcaps'
os.makedirs(DIR, exist_ok=True)

url = 'https://mergetb.org/projects/searchlight/dataset/'
print('Acquiring links...')
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.pcap')]

keywords = ['ptp', 'tiered', 'video', 'dash']
filtered_links = [link for link in links if all(keyword in link for keyword in keywords)]

for link in filtered_links:
    filename = link.split('/')[-1]
    filepath = os.path.join(DIR, filename)
    print(f'Downloading {filename}...')
    response = requests.get(link)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    print(f'{filename} downloaded to {filepath}')