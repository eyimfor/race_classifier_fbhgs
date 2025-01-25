# race_classifier_fbhgs
# Race Classifier for "Funding Black High-Growth Startups"

This repository contains the initial race classification algorithm used in "Funding Black High-Growth Startups" (Cook, Marx, and Yimfor, forthcoming in Journal of Finance). The classifier, which combines facial recognition technology (DeepFace) with Census surname data, served as a first-pass screening tool. All classifications were subsequently reviewed manually to ensure accuracy, with disagreements resolved through additional review. The final race classifications in the paper reflect this thorough clerical review process rather than solely algorithmic predictions.

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

## Citation
```bibtex
@article{yimfor2025funding,
title={Funding Black High-Growth Startups},
author={Yimfor, Emmanuel and Marx, Matt and Cook, Lisa D},
journal={Journal of Finance},
note={Forthcoming},
year={2025}
}
