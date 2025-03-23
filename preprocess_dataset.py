import os
import cv2
import numpy as np
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMAGE_SIZE = (224, 224)
DATASET_PATH = r"C:\Users\Anshu Kumar Rajak\Desktop\Skin\skin-disease-detection\dataset"
PROCESSED_DATA_PATH = os.path.join(DATASET_PATH, "processed_data")
AUGMENTATION_LIMIT = 5  # Number of augmented images per original image

# Create the processed data directory if it doesn't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

def preprocess_and_augment_image(image_path, save_path, augmentation_limit=AUGMENTATION_LIMIT):
    """Reads an image, preprocesses it, and applies augmentation."""
    image = cv2.imread(image_path)

    if image is None:
        logging.warning(f"Unable to read image at {image_path}. Skipping this file.")
        return

    # Resize the image
    image = cv2.resize(image, IMAGE_SIZE)
    image = np.expand_dims(image, axis=0)  # Reshape to (1, 224, 224, 3)

    # Generate augmented images
    augmented_images = datagen.flow(image, batch_size=1)

    for i in range(augmentation_limit):
        augmented_image = next(augmented_images)[0]
        augmented_image = (augmented_image * 255).astype(np.uint8)

        # Save augmented image
        augmented_image_path = os.path.join(save_path, f"augmented_{i}.jpg")
        cv2.imwrite(augmented_image_path, augmented_image)
        logging.info(f"Saved augmented image: {augmented_image_path}")

# Process each category (class) in the dataset
for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)

    # Skip if not a directory or is the processed_data folder
    if not os.path.isdir(category_path) or category == "processed_data":
        logging.info(f"Skipping {category_path}")
        continue

    logging.info(f"Processing category: {category}")

    # Create a folder inside processed_data for this category
    save_category_path = os.path.join(PROCESSED_DATA_PATH, category)
    os.makedirs(save_category_path, exist_ok=True)

    # Process each image file in the category folder
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)

        # Skip non-image files
        if not os.path.isfile(image_path) or not image_name.lower().endswith((".jpg", ".png", ".jpeg")):
            logging.info(f"Skipping non-image file: {image_path}")
            continue

        logging.info(f"Processing image: {image_path}")
        preprocess_and_augment_image(image_path, save_category_path)

logging.info("✅ Data preprocessing and augmentation completed successfully!")









# first code 

# import os
# import cv2
# import numpy as np
# import logging
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Define constants
# IMAGE_SIZE = (224, 224)  # Resize images to 224x224
# DATASET_PATH = r"C:\Users\Anshu Kumar Rajak\Desktop\Skin\skin-disease-detection\dataset"
# PROCESSED_DATA_PATH = os.path.join(DATASET_PATH, "processed_data")
# AUGMENTATION_LIMIT = 5  # Number of augmented images per original image

# # Ensure processed_data directory exists
# os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Initialize ImageDataGenerator for augmentation
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode="nearest",
# )


# def preprocess_and_augment_image(image_path, save_path, augmentation_limit=AUGMENTATION_LIMIT):
#     """Reads an image, preprocesses it, and applies augmentation."""
#     logging.info(f"Reading image: {image_path}")

#     # Read the image
#     image = cv2.imread(image_path)
#     if image is None:
#         logging.warning(f"Unable to read image: {image_path}. Skipping.")
#         return

#     # Resize image to match the model's expected input size
#     image = cv2.resize(image, IMAGE_SIZE)

#     # Convert image to float32 and normalize (Keras expects float data)
#     image = image.astype("float32") / 255.0

#     # Reshape image to fit Keras expectations: (1, 224, 224, 3)
#     image = np.expand_dims(image, axis=0)

#     # Create augmented images
#     augmented_images = datagen.flow(image, batch_size=1)

#     for i in range(augmentation_limit):
#         augmented_image = next(augmented_images)[0]  # Get the first image from batch
#         augmented_image = (augmented_image * 255).astype(np.uint8)  # Convert back to uint8

#         # Save augmented image
#         augmented_image_path = os.path.join(save_path, f"augmented_{i}.jpg")
#         cv2.imwrite(augmented_image_path, augmented_image)
#         logging.info(f"Saved augmented image: {augmented_image_path}")


# # Process each category (class) in the dataset
# for category in os.listdir(DATASET_PATH):
#     category_path = os.path.join(DATASET_PATH, category)

#     # Skip if not a directory or is the processed_data folder
#     if not os.path.isdir(category_path) or category == "processed_data":
#         logging.info(f"Skipping {category_path}")
#         continue

#     logging.info(f"Processing category: {category}")

#     # Create a folder inside processed_data for this category
#     save_category_path = os.path.join(PROCESSED_DATA_PATH, category)
#     os.makedirs(save_category_path, exist_ok=True)

#     # Process each image file in the category folder
#     for image_name in os.listdir(category_path):
#         image_path = os.path.join(category_path, image_name)

#         # Skip non-image files
#         if not os.path.isfile(image_path) or not image_name.lower().endswith((".jpg", ".png", ".jpeg")):
#             logging.info(f"Skipping non-image file: {image_path}")
#             continue

#         logging.info(f"Processing image: {image_path}")
#         preprocess_and_augment_image(image_path, save_category_path)

# logging.info("✅ Data preprocessing and augmentation completed successfully!")