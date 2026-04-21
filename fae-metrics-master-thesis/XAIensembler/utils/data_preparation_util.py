import os
import shutil
import pandas as pd

segmentation_data = True

csv_file = "PATH" # Update this path

image_directory = "PATH" # Update this path

df = pd.read_csv(csv_file)

# Create category directories inside the image directory
categories = ["melanoma", "seborrheic_keratosis", "nevus"]
category_paths = {category: os.path.join(image_directory, category) for category in categories}

for path in category_paths.values():
    os.makedirs(path, exist_ok=True)

for _, row in df.iterrows():

    image_name = f"{row['image_id']}.jpg" if segmentation_data == False else f"{row['image_id']}_segmentation.png"
    source_path = os.path.join(image_directory, image_name)

    if os.path.exists(source_path):
        for category in categories:
            if row[category] == 1:
                destination_path = os.path.join(category_paths[category], image_name)
                shutil.copy(source_path, destination_path)
                print(f"Copied {image_name} to {category_paths[category]}")

print("Image sorting completed.")
