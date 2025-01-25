# race_classifier_fbhgs
# Race Classifier for "Funding Black High-Growth Startups"

This repository contains both the race classification code and dataset from "Funding Black High-Growth Startups" (Yimfor, Marx, and Cook, forthcoming in Journal of Finance). The dataset covers U.S.-based startups founded between 2000-2020 with founder information from PitchBook, merged with SEC Form D filings. The classifier combines facial recognition technology (DeepFace) with Census surname data to predict founders' race. This served as a first-pass screening tool. All classifications were subsequently reviewed manually by multiple research assistants to ensure accuracy.

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

### Citation
```bibtex
@article{yimfor2025funding,
title={Funding Black High-Growth Startups},
author={Yimfor, Emmanuel and Marx, Matt and Cook, Lisa D},
journal={Journal of Finance},
note={Forthcoming},
year={2025}
}
