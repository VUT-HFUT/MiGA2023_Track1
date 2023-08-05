#coding:utf-8

import os
import pickle
import numpy as np
import pandas as pd
import numpy as np
import zipfile
from os.path import basename
import tqdm

def read_numpy(file_path):
    return np.load(file_path)

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data


with open('./test_id.txt', 'r') as f:
    test_id = f.read()
    f.close()

test_id = list(test_id.split(','))

###################################
# weighted average 
pred1 = load_pkl('./posec3d_emb20_joint/result.pkl')
pred2 = load_pkl('./posec3d_emb20_limb/result.pkl')

weight1 = 0.4
weight2 = 0.6

ensemble_pred = weight1 * np.array(pred1) + weight2 * np.array(pred2)
out_rank = np.argmax(ensemble_pred, axis=1)

###################################
# create sumission
predPath = './'
if not os.path.exists(predPath):
    os.makedirs(predPath)
output_filename = os.path.join(predPath, 'Submission.csv')
output_file = open(output_filename, 'w')

for inx in tqdm.tqdm(range(len(test_id))):
    output_file.write(str(test_id[inx]) + "," + str(out_rank[inx]) + "\n")
output_file.close()

# Set the name of the output zip file
zip_file_name = os.path.join(predPath, 'Submission.zip')

# Create a ZipFile object with the output zip file name and mode
with zipfile.ZipFile(zip_file_name, "w") as zip_file:
    # Add the file you want to zip to the archive
    zip_file.write(output_filename, basename(output_filename))
