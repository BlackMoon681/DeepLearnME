import os
import shutil
import kagglehub

# Project root directory
project_root = "C:\\Users\\mcf\\PycharmProjects\\DeepLearnME"


# Download and extract dataset
def download_dataset():
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created data/ directory.")
    else:
        print("data/ directory already exists. Proceeding with download.")

    # Download dataset using kagglehub
    try:
        dataset_path = kagglehub.dataset_download("jockeroika/human-bone-fractures-image-dataset")
        print("Dataset downloaded to:", dataset_path)

        # Move dataset contents to data/ if nested
        if os.path.basename(dataset_path) == "human-bone-fractures-image-dataset":
            for item in os.listdir(dataset_path):
                src_path = os.path.join(dataset_path, item)
                dest_path = os.path.join(data_dir, item)
                if os.path.isdir(src_path):
                    shutil.move(src_path, dest_path)
            # Remove the empty downloaded folder
            shutil.rmtree(dataset_path)
        else:
            # If not nested, move the entire folder
            dest_path = os.path.join(data_dir, os.path.basename(dataset_path))
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)  # Remove if already exists to avoid conflicts
            shutil.move(dataset_path, data_dir)
        print("Dataset structure adjusted in data/.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure kagglehub is installed (pip install kagglehub) and you have an internet connection.")
        print("You may need a Kaggle API token configured if authentication is required.")


if __name__ == "__main__":
    # Download and add dataset
    download_dataset()

    print("Setup complete! Check the data/ folder for the dataset.")
