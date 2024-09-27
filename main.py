import os
import uuid
import pandas as pd
from preference_collection import collect_preference
from dpo_training import optimize_model_with_dpo

# Paths for data and model storage
DATA_DIR = "./data"
MODELS_DIR = "./models"
DEBUG_DIR = "./debug"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Helper function to check if a file exists
def file_exists(step_name):
    """Checks if a file for a particular step exists in the debug folder."""
    for filename in os.listdir(DEBUG_DIR):
        if step_name in filename:
            return os.path.join(DEBUG_DIR, filename)
    return None

# Debug function to write intermediate data with UUID
def write_to_debug_folder(data, step_name):
    """Writes data to a debug folder with a unique UUID."""
    unique_id = str(uuid.uuid4())
    filename = os.path.join(DEBUG_DIR, f"{step_name}_{unique_id}.csv")
    data.to_csv(filename, index=False)
    print(f"Debug: Data for {step_name} saved to {filename}.")
    return filename  # Return the filename

# 1. Preference Collection Function
def run_preference_collection(data):
    print("Running Preference Collection...")

    # Check if the preferences file already exists
    preferences_file = file_exists("preferences")
    if preferences_file:
        print(f"Found existing preferences dataset at: {preferences_file}")
        return pd.read_csv(preferences_file)
    
    # Collect preferences (human or AI feedback)
    preference_data = collect_preference(data)
    
    # Save the preferences dataset to the debug folder
    preferences_filename = write_to_debug_folder(preference_data, "preferences")
    print(f"Preference Collection completed. Dataset saved with preferences to: {preferences_filename}")
    return preference_data

# 2. DPO Training Function
def run_dpo_training(preference_data):
    print("Running DPO Training...")

    # Check if the model is already optimized
    optimized_model_file = file_exists("dpo_model")
    if optimized_model_file:
        print(f"Found existing optimized model at: {optimized_model_file}")
        return
    
    # Optimize the model based on the collected preferences
    optimize_model_with_dpo(preference_data)

    # Save the updated model to the models folder
    print(f"DPO Training completed. Model saved to the models directory.")

# Function that implements the entire pipeline
def run_pipeline(data):
    # Step 1: Preference Collection
    preference_data = run_preference_collection(data)

    # Step 2: Optimize the Model using DPO
    run_dpo_training(preference_data)

    print("DPO pipeline completed successfully.")

# Main function
def main():
    print("Reading dataset...")

    # Load the dataset from the data folder
    data = pd.read_csv(f"{DATA_DIR}/dataset.csv")
    
    # Run the pipeline with the dataset
    run_pipeline(data)

if __name__ == "__main__":
    main()
