# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os

# # Define constants
# IMAGE_SIZE = (224, 224)
# BATCH_SIZE = 32
# EPOCHS = 20
# DATASET_PATH = r"C:\Users\Anshu Kumar Rajak\Desktop\Skin\skin-disease-detection\dataset"

# # Check if the dataset directory exists
# if not os.path.exists(DATASET_PATH):
#     raise FileNotFoundError(f"Dataset directory not found at {DATASET_PATH}. Please check the path and try again.")



# # Verify the dataset directory structure
# print("Verifying dataset directory structure...")
# def count_files_in_directory(directory):
#     total_files = 0
#     with os.scandir(directory) as entries:
#         for entry in entries:
#             if entry.is_file():
#                 total_files += 1
#             elif entry.is_dir():
#                 total_files += count_files_in_directory(entry.path)
#     return total_files

# # print("Verifying dataset directory structure...")
# # total_files = count_files_in_directory(DATASET_PATH)
# # print(DATASET_PATH, "contains", total_files, "files")

# # Initialize ImageDataGenerator for training and validation
# datagen = ImageDataGenerator(validation_split=0.2)

# train_generator = datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# # Check if images are found
# if train_generator.samples == 0:
#     raise ValueError(f"No images found in the dataset directory at {DATASET_PATH}. Please check the directory structure and paths.")

# validation_generator = datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# # Check if images are found
# if validation_generator.samples == 0:
#     raise ValueError(f"No images found in the dataset directory at {DATASET_PATH}. Please check the directory structure and paths.")

# # Create the advanced CNN model
# model = Sequential([
#     Input(shape=(*IMAGE_SIZE, 3)),
#     Conv2D(32, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D((2, 2)),
#     Dropout(0.25),
    
#     Conv2D(64, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D((2, 2)),
#     Dropout(0.25),
    
#     Conv2D(128, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D((2, 2)),
#     Dropout(0.25),
    
#     Flatten(),
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.5),
#     Dense(train_generator.num_classes, activation='softmax')
# ])

# # Compile the model
# learning_rate = 0.001  # Specify the learning rate
# model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(
#     train_generator,
#     epochs=EPOCHS,
#     validation_data=validation_generator
# )

# # Save the model
# model_save_path = r"C:\Users\Anshu Kumar Rajak\Desktop\Skin\skin-disease-detection\models\advanced_skin_disease_detection_model.h5"
# model.save(model_save_path)
# print(f"Model saved successfully at {model_save_path}.")


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = r'C:\Users\Anshu Kumar Rajak\Desktop\Skin\skin-disease-detection\models\advanced_skin_disease_detection_model.h5'
DATASET_PATH = r"C:\Users\Anshu Kumar Rajak\Desktop\Skin\skin-disease-detection\dataset"

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load the VGG16 model with pre-trained weights, excluding the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=EPOCHS
)

# Save the model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
