import os
import subprocess
import json
from PIL import Image
import pandas as pd
import numpy as np

def ip_to_binary(ip_address):
    # Split the IP address into four octets
    octets = ip_address.split(".")
    # Convert each octet to binary form and pad with zeros
    binary_octets = []
    for octet in octets:
        binary_octet = bin(int(octet))[2:].zfill(8)
        binary_octets.append(binary_octet)

    # Concatenate the four octets to get the 32-bit binary representation
    binary_ip_address = "".join(binary_octets)
    return binary_ip_address
def binary_to_ip(binary_ip_address):
    # Check that the input binary string is 32 bits
    if len(binary_ip_address) != 32:
        raise ValueError("Input binary string must be 32 bits")
    # Split the binary string into four octets
    octets = [binary_ip_address[i:i+8] for i in range(0, 32, 8)]
    # Convert each octet from binary to decimal form
    decimal_octets = [str(int(octet, 2)) for octet in octets]

    # Concatenate the four decimal octets to get the IP address string
    ip_address = ".".join(decimal_octets)
    return ip_address

# Convert the DataFrame into a PNG image
def dataframe_to_png(df, output_file):
    width, height = df.shape[1], df.shape[0]
    padded_height = 1024
    print(output_file)
    # Convert DataFrame to numpy array and pad with blue pixels
    np_img = np.full((padded_height, width, 4), (0, 0, 255, 255), dtype=np.uint8)
    np_df = np.array(df.apply(lambda x: x.apply(np.array)).to_numpy().tolist())
    np_img[:height, :, :] = np_df

    # Create a new image with padded height filled with blue pixels
    img = Image.fromarray(np_img, 'RGBA')

    # Check if file exists and generate a new file name if necessary
    file_exists = True
    counter = 1
    file_path, file_extension = os.path.splitext(output_file)
    while file_exists:
        if os.path.isfile(output_file):
            output_file = f"{file_path}_{counter}{file_extension}"
            counter += 1
        else:
            file_exists = False

    img.save(output_file)


def rgba_to_ip(rgba):
    ip_parts = tuple(map(str, rgba))
    ip = '.'.join(ip_parts)
    return ip
    
def int_to_rgba(A):
    if A == 1:
        rgba = (255, 0, 0, 255)
    elif A == 0:
        rgba = (0, 255, 0, 255)
    elif A == -1:
        rgba = (0, 0, 255, 255)
    elif A > 1:
        rgba = (255, 0, 0, A)
    elif A < -1:
        rgba = (0, 0, 255, abs(A))
    else:
        rgba = None
    return rgba

def rgba_to_int(rgba):
    if rgba == (255, 0, 0, 255):
        A = 1
    elif rgba == (0, 255, 0, 255):
        A = 0
    elif rgba == (0, 0, 255, 255):
        A = -1
    elif rgba[0] == 255 and rgba[1] == 0 and rgba[2] == 0:
        A = rgba[3]
    elif rgba[0] == 0 and rgba[1] == 0 and rgba[2] == 255:
        A = -rgba[3]
    else:
        A = None
    return A

# define function to split binary string into individual bits
def split_bits(s):
    return [int(b) for b in s]

data_dir = '../data'
pcap_dir = f"{data_dir}/full_pcaps"
full_nprint_dir = f"{data_dir}/full_nprints"
finetune_nprint_dir = f"{data_dir}/finetune_nprints"
img_dir = f"{data_dir}/finetune_imgs"
os.makedirs(os.path.dirname(img_dir), exist_ok=True)

output = f"{data_dir}/captions.json"
captions = {file.split('-')[0]: f"{file.split('-')[-3]}, {file.split('.')[-4]}, {file.split('.')[-2]}" for file in os.listdir(pcap_dir)}

with open (output, 'w') as f:
	json.dump(captions, f)

for file in os.listdir(pcap_dir):
    print(file)
    filename = file.split('.pcap')[0]
    pcap_filename = f"{pcap_dir}/{file}"
    full_nprint_filename = f"{full_nprint_dir}/{filename}.nprint"
    finetune_nprint_filename = f"{finetune_nprint_dir}/{filename}.nprint"
    print(f"creating nprint for {pcap_filename}")
    subprocess.Popen(f"nprint -F -1 -P {pcap_filename} -4 -i -6 -t -u -p 0 -W {full_nprint_filename}", shell=True)
    subprocess.Popen(f"nprint -F -1 -P {pcap_filename} -4 -i -6 -t -u -p 0 -c 1024 -W {finetune_nprint_filename}", shell=True).wait()
    print('done.')

    nprint_path = f"{finetune_nprint_dir}/{filename}.nprint"
    df = pd.read_csv(nprint_path)
    num_packet = df.shape[0]
    if num_packet != 0:
        try:
            substrings = ['ipv4_src', 'ipv4_dst', 'ipv6_src', 'ipv6_dst','src_ip']
            cols_to_drop = [col for col in df.columns if any(substring in col for substring in substrings)]
            df = df.drop(columns=cols_to_drop)
            cols = df.columns.tolist()
            for col in cols:
                df[col] = df[col].apply(int_to_rgba)

            print(df.shape)
            output_file = f"{img_dir}/{filename}.png"
            dataframe_to_png(df, output_file)
        except Exception as e:
            print(e)
            continue