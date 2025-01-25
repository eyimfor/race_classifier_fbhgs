# race_classifier_fbhgs
# Race Classifier for "Funding Black High-Growth Startups"

This repository contains the initial race classification algorithm used in "Funding Black High-Growth Startups" (Yimfor, Marx, and Cook, forthcoming in Journal of Finance). The classifier combines facial recognition technology (DeepFace) with Census surname data to predict founders' race. This served as a first-pass screening tool. All classifications were subsequently reviewed manually by multiple research assistants to ensure accuracy.

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

## Usage
```bash
python race_classifier_fbhgs.py <input_folder> <output_folder>

## Citation
```bibtex
@article{yimfor2025funding,
title={Funding Black High-Growth Startups},
author={Yimfor, Emmanuel and Marx, Matt and Cook, Lisa D},
journal={Journal of Finance},
note={Forthcoming},
year={2025}
}
