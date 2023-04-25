""" 
Sample images and reports from `raw/` into a train-valid-test split, ready for the data loader.
"""

import pandas as pd
from brainclip.config import *

parsed_reports_df = pd.read_csv(parsed_reports)

# scp -r /datadrive_m2/marko/data/Alice_IN-BodyScanData-03 /datadrive_m2/alice/brain-CLIP/data/raw/images