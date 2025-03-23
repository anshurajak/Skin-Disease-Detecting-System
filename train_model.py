import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

# ✅ Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25  # Initial Training
FINE_TUNE_EPOCHS = 10  # Fine-tuning
DATASET_PATH = r'C:\Users\Anshu Kumar Rajak\Desktop\Skin\skin-disease-detection\dataset'
MODEL_SAVE_PATH = r'C:\Users\Anshu Kumar Rajak\Desktop\Skin\skin-disease-detection\models\advanced_skin_disease_detection_model.h5'

# ✅ Ensure model save directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# ✅ Load dataset BEFORE prefetching to extract class names
raw_train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

raw_val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# ✅ Extract class names BEFORE prefetching
class_names = raw_train_dataset.class_names
num_classes = len(class_names)
print("\n📊 Class Indices:", class_names)

# ✅ Apply prefetch for performance boost
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = raw_train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = raw_val_dataset.prefetch(buffer_size=AUTOTUNE)

# ✅ Compute Class Weights (for Imbalance Handling)
labels = np.concatenate([y.numpy() for _, y in raw_train_dataset])
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# ✅ Build Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model for initial training

model = Sequential([
    Input(shape=(224, 224, 3)),
    base_model,
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    
    GlobalAveragePooling2D(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# ✅ Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# ✅ Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

# ✅ Train Model
print("\n🚀 Training Model...")
model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, 
          class_weight=class_weight_dict, callbacks=callbacks)

# ✅ Fine-Tune Model
print("\n🔓 Unfreezing last 4 layers for fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:-4]:  
    layer.trainable = False

# ✅ Recompile and Fine-Tune
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, epochs=FINE_TUNE_EPOCHS, validation_data=val_dataset, 
          class_weight=class_weight_dict, callbacks=callbacks)

# ✅ Save the Model
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model saved successfully at: {MODEL_SAVE_PATH}")

# ✅ Evaluate Model
val_loss, val_acc = model.evaluate(val_dataset)
print(f"\n📊 Final Validation Accuracy: {val_acc:.4f}")
print(f"📉 Final Validation Loss: {val_loss:.4f}")                                  #This code in proper working condition
