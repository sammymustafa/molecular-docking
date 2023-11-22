import scanpy as sc
import anndata
import os
import pandas as pd
import scipy.io as io
import numpy as np
from scipy.spatial.transform import Rotation
from rdkit import Chem
from rdkit.Chem import AllChem

# Downloading and extracting data
import zipfile
import requests

def download_and_unzip(url, extract_to='.'):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Download and extract data
download_and_unzip("https://www.dropbox.com/sh/4f9fmenyvmrffj2/AAAmNwJtO8ZKTaRkZoltTCHsa?dl=1", "data")
download_and_unzip("https://www.dropbox.com/scl/fi/e23wyl4bfolckjaaablhc/diffdock.zip?dl=1", "diffdock")