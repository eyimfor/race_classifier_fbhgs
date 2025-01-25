import hashlib                     
import os                          
import csv                         
import tqdm                        
import pandas as pd                
import pickle                     
import shutil                      
from deepface import DeepFace      
from ethnicolr import census_ln    
import numpy as np

def imageClassify_race(image_path: str, output_folder: str) -> None:
    """
    Classifies images by race using facial recognition and surname analysis.
    
    Args:
        image_path: Directory containing images named as 'Firstname_Lastname_ID.jpg'
        output_folder: Base directory for output folders (one per race category)
        
    Note: Requires yimfor_random_forest_model.sav in working directory
    """
    def get_sha256_hash(file_path: str) -> str:
        """Generate unique hash for image identification."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()

    # Step 1: Create image hash mapping
    failed = []
    with open('../image_to_hash.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Hash', 'Image Path', 'Image_Name'])
        for filename in tqdm.tqdm(os.listdir(image_path), desc="Building hash map"):
            try:
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(image_path, filename)
                    hash_value = get_sha256_hash(file_path)
                    writer.writerow([hash_value, file_path, filename])
            except Exception as e:
                failed.append(filename)
                continue
    print("Finished building Hash map for images")
    
    processed_images = pd.read_csv('../image_to_hash.csv', low_memory=False)

    # Step 2: DeepFace Analysis
    def deepfaceraceproba(row):
        """Analyze racial features in image using DeepFace."""
        try:
            obj = DeepFace.analyze(img_path=row['Image Path'], 
                                 actions=['race'], 
                                 enforce_detection=False, 
                                 silent=True)
            data = pd.DataFrame(obj[0]['race'], index=[1])
            data['Hash'] = row['Hash']
            data.to_csv('../DeepFace.csv', index=False, header=False, mode='a')
        except Exception:
            return

    # Initialize DeepFace results file with first image
    obj = DeepFace.analyze(img_path=processed_images.loc[0, 'Image Path'], 
                          actions=['race'], 
                          enforce_detection=False, 
                          silent=True)
    data = pd.DataFrame(obj[0]['race'], index=[1])
    data['Hash'] = processed_images.loc[0, 'Hash']
    data.to_csv('../DeepFace.csv', index=False)

    # Process remaining images
    deepfaceraceproba(processed_images.loc[1])
    for row in tqdm.tqdm(processed_images.index, desc="Running DeepFace analysis"):
        deepfaceraceproba(processed_images.loc[row])
    print("Finished running Deep Face")

    # Step 3: Census Analysis
    processed_names = pd.read_csv("../image_to_hash.csv", encoding='latin-1')
    processed_names['lastname'] = processed_names['Image_Name'].apply(lambda x: x.split("_")[1])

    last_names = processed_names[['Hash', 'lastname']].drop_duplicates()
    last_names.columns = ['Hash', 'lastname']
    prediction_census = census_ln(last_names, 'lastname')
    
    # Select relevant census columns
    census_columns = ['Hash', 'pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']
    prediction_census = prediction_census[census_columns].drop_duplicates()
    prediction_census.replace('(S)', 0, inplace=True)
    
    # Merge census data
    census = processed_names.merge(prediction_census, on='Hash', how='left')
    census = census[census_columns].drop_duplicates()

    # Step 4: Combine DeepFace and Census predictions
    deepface = pd.read_csv("../DeepFace.csv", on_bad_lines='skip', low_memory=False)
    prediction_combined = deepface.merge(census, on='Hash', how='inner')

    # Handle missing predictions
    namecolumns = ['pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']
    prediction_combined['no_prediction'] = prediction_combined[namecolumns].isna().all(axis=1).astype(int)

    # Prepare features for model
    feature_cols = ['asian', 'indian', 'black', 'white', 'middle eastern',
                   'latino hispanic', 'pctwhite', 'pctblack', 'pctapi', 'pctaian',
                   'pct2prace', 'pcthispanic', 'no_prediction']
    
    prediction_combined = prediction_combined[['Hash'] + feature_cols].drop_duplicates()
    prediction_combined = prediction_combined.set_index('Hash')

    # Clean census data
    for column in namecolumns:
        prediction_combined[column] = pd.to_numeric(prediction_combined[column], errors='coerce')
        prediction_combined.loc[prediction_combined[column].isna(), column] = 0
        prediction_combined.loc[prediction_combined[column].isna(), 'no_prediction'] = 1

    # Step 5: Random Forest Prediction
    with open(r"yimfor_random_forest_model.sav", "rb") as input_file:
        model = pickle.load(input_file)

    prediction_combined = prediction_combined[feature_cols].drop_duplicates()
    predictions = model.predict(prediction_combined)
    
    prediction_combined = prediction_combined.reset_index()
    prediction_combined['final_race'] = predictions
    prediction_combined = prediction_combined[['Hash', 'final_race']].drop_duplicates()

    # Map numerical predictions to race labels
    mapping = {3: 'white', 2: 'hispanic', 1: 'black', 0: 'asian'}
    prediction_combined["racelabel"] = prediction_combined["final_race"].map(mapping)
    print("Finished building race prediction")

    # Step 6: Organize images into race folders
    processed_images = pd.read_csv('../image_to_hash.csv', low_memory=False)
    relevant_inputs = prediction_combined.merge(processed_names, on='Hash', how='inner')

    for index, row in tqdm.tqdm(relevant_inputs.iterrows(), desc="Organizing images"):
        race_folder = output_folder + row['racelabel'].lower()
        if not os.path.exists(race_folder):
            os.makedirs(race_folder)
            
        shutil.copy2(row['Image Path'], os.path.join(race_folder, row['Image_Name']))
    print("Finished getting images ready for clerical review")
    
    # Cleanup temporary files
    os.remove('../image_to_hash.csv')
    os.remove('../DeepFace.csv')

    print("DISCLAIMER: Results require manual verification for accuracy.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python race_classifier.py <input_path> <output_path>")
        sys.exit(1)
    imageClassify_race(sys.argv[1], sys.argv[2])