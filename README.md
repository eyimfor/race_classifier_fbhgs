# race_classifier_fbhgs
# Race Classifier for "Funding Black High-Growth Startups"

This repository contains both the race classification code and dataset from "Funding Black High-Growth Startups" (Cook, Marx, and Yimfor, Journal of Finance, 2026). The dataset covers U.S.-based startups founded between 2000-2020 with founder information from PitchBook, merged with SEC Form D filings. The classifier combines facial recognition technology (DeepFace) with Census surname data to predict founders' race. This served as a first-pass screening tool. All classifications were subsequently reviewed manually by multiple research assistants to ensure accuracy.

## Version 2 of the classifier
A newer classifier is in [`race_classifier_v2/`](race_classifier_v2/). It reads first and last name together, builds on FairFace for the face, and is much more accurate on Black, Hispanic, and Asian founders. The original classifier below is the one used in the paper and is unchanged.

## Setup
1. Install required packages: `pip install -r requirements.txt`
2. Download `yimfor_random_forest_model.zip` and extract `yimfor_random_forest_model.sav` to the same directory as the code

## Features
- Processes images named as 'Firstname_Lastname_ID'
- Combines DeepFace facial analysis with Census surname data
- Uses Random Forest model for final classification
- Outputs sorted images into race-specific folders

## Requirements
- Python 3.7+
- DeepFace
- ethnicolr
- pandas
- numpy
- scikit-learn

## File Contents
`Funding_Black_High-Growth_Startups_DataSet_09_30_2024.xlsx` contains:
- `cik`: Form D filer unique identifier
- `formdfilingurl`: Link to Form D filing
- `entityname`: Startup name from Form D
- `nameformd`: Founder name from Form D
- `std_url`: Founder's LinkedIn URL

## Data Collection
Sample constructed from:
1. PitchBook data on U.S. startups (2000-2020)
2. Profile images from public sources
3. SEC Form D filings matched by firm name/location
4. Founder race classification using:
  - DeepFace facial analysis
  - Name analysis
  - Manual verification of all Black founder classifications

    
## Usage
```bash
python race_classifier_fbhgs.py <input_folder> <output_folder>
```

### Citation
```bibtex
@article{cook2026funding,
title={Funding Black High-Growth Startups},
author={Cook, Lisa D. and Marx, Matt and Yimfor, Emmanuel},
journal={The Journal of Finance},
volume={81},
number={3},
pages={1619--1660},
year={2026},
doi={10.1111/jofi.70039}
}
```
