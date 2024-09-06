import os
import json
from PIL import Image
import matplotlib.pyplot as plt

# Folder containing images
image_folder = 'C:/Users/SPEREPA1/metapix-delivery-day-tool/car_images'
# File to save labels
label_file = 'C:/Users/SPEREPA1/metapix-delivery-day-tool/image_labels.json'

def display_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def label_images(image_folder):
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    labels = {}

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        display_image(image_path)
        
        # Ask user for labels
        make = input(f"Enter the make of the car for {image_file}: ")
        model = input(f"Enter the model of the car for {image_file}: ")
        color = input(f"Enter the color of the car for {image_file}: ")
        description = input(f"Enter additional description for {image_file} (optional): ")

        labels[image_file] = {
            'make': make,
            'model': model,
            'color': color,
            'description': description
        }
    
    # Save labels to a JSON file
    with open(label_file, 'w') as f:
        json.dump(labels, f, indent=4)

    print(f"Labels saved to {label_file}")

if __name__ == '__main__':
    label_images(image_folder)
