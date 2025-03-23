#1 code if this code is not working properly go to 2 code

# import numpy as np
# import tensorflow as tf
# import cv2
# import matplotlib.pyplot as plt
# import logging
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from detection_app.model_utils import load_trained_model, preprocess_image, predict_skin_disease

# # Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# # Load trained model
# model = load_trained_model()
# if model is None:
#     raise RuntimeError("❌ Model loading failed. Cannot proceed!")

# # Ensure model is initialized
# _ = model.predict(np.zeros((1, 224, 224, 3)))  # Dummy prediction

# # Unfreeze model layers to allow gradient computation
# model.trainable = True
# for layer in model.layers:
#     layer.trainable = True

# # Get the last Conv2D layer                                               #*This code in proper working condition*
# def get_last_conv_layer(model):
#     """Finds the last Conv2D layer in the model."""
#     for layer in reversed(model.layers):
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             return layer.name
#     raise ValueError("❌ No Conv2D layer found in the model!")

# last_conv_layer_name = get_last_conv_layer(model)
# logging.info(f"✅ Last Conv2D Layer: {last_conv_layer_name}")

# # Grad-CAM Function
# def grad_cam(model, img, class_index, conv_layer_name):
#     """Generate Grad-CAM heatmap for a given image and class index."""
    
#     class_index = int(class_index)  # Ensure class_index is an integer

#     # Define a model that outputs both the feature maps and predictions
#     grad_model = tf.keras.models.Model(
#         inputs=model.inputs,
#         outputs=[model.get_layer(conv_layer_name).output, model.outputs[0]]
#     )

#     # Ensure img has the correct shape (1, 224, 224, 3)
#     if len(img.shape) == 4:
#         img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)  # Image already batched
#     else:
#         img_tensor = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)  # Expand batch dim

#     logging.debug(f"Image tensor shape before Grad-CAM: {img_tensor.shape}")

#     with tf.GradientTape() as tape:
#         tape.watch(img_tensor)  # Ensure TensorFlow tracks it
#         conv_outputs, predictions = grad_model(img_tensor, training=False)  # Ensure training=False

#         logging.debug(f"conv_outputs shape: {conv_outputs.shape}")
#         logging.debug(f"predictions shape: {predictions.shape}")

#         loss = predictions[:, class_index]  # Extract loss for the target class

#     # Compute gradients
#     grads = tape.gradient(loss, conv_outputs)
#     if grads is None:
#         raise ValueError("❌ Gradient computation failed. Ensure the model's output is correct.")
    
#     logging.debug(f"grads shape: {grads.shape}")

#     # Compute pooled gradients
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     logging.info(f"pooled_grads shape: {pooled_grads.shape}")

#     # Apply weights to feature maps using broadcasting
#     conv_outputs = conv_outputs[0]  # Remove batch dimension
#     conv_outputs = conv_outputs * pooled_grads[None, None, :]

#     # Generate heatmap
#     heatmap = np.mean(conv_outputs, axis=-1)
#     heatmap = np.maximum(heatmap, 0)  # ReLU
#     heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1  # Normalize (avoid division by zero)

#     return heatmap

# # Overlay Heatmap on Image
# def overlay_grad_cam(image_path, heatmap):
#     """Overlay Grad-CAM heatmap on the original image."""
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (224, 224))

#     heatmap = cv2.resize(heatmap, (224, 224))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     overlayed_image = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

#     # Show results
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.title("Original Image")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(overlayed_image)
#     plt.title(f"{predicted_label}")
#     plt.axis("off")

#     plt.show()

# # Main Execution
# if __name__ == "__main__":
#     IMAGE_PATH = r'C:\Users\Anshu Kumar Rajak\Desktop\Skin\skin-disease-detection\dataset\acne\image1.png'

#     # Predict Skin Disease
#     prediction = predict_skin_disease(model, IMAGE_PATH)
#     if prediction is None or prediction[0] == "Uncertain":
#         logging.warning("⚠️ Uncertain prediction. Grad-CAM not generated.")
#     else:
#         predicted_label, class_index = prediction

#         # Preprocess image
#         img_array = preprocess_image(IMAGE_PATH)

#         # Ensure the model is "called" before Grad-CAM
#         _ = model(img_array, training=False)

#         # Generate Grad-CAM heatmap
#         heatmap = grad_cam(model, img_array, class_index, last_conv_layer_name)

#         logging.info(f"Predicted Disease: {predicted_label}")
#         overlay_grad_cam(IMAGE_PATH, heatmap)

#         # Print the predicted label
#         print(f"Predicted Disease: {predicted_label}")


#2 code work properly
# import numpy as np
# import tensorflow as tf
# import cv2
# import matplotlib.pyplot as plt
# import logging
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'skin-disease-detection')))

# from detection_app.model_utils import load_trained_model, preprocess_image, predict_skin_disease

# # Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# # Load trained model
# model = load_trained_model()
# if model is None:
#     raise RuntimeError("❌ Model loading failed. Cannot proceed!")

# # Ensure model is initialized
# _ = model.predict(np.zeros((1, 224, 224, 3)))  # Dummy prediction

# # Ensure model is trainable for Grad-CAM
# model.trainable = True
# for layer in model.layers:
#     layer.trainable = True

# # Get the last Conv2D layer
# def get_last_conv_layer(model):
#     for layer in reversed(model.layers):
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             return layer.name
#     raise ValueError("❌ No Conv2D layer found in the model!")

# last_conv_layer_name = get_last_conv_layer(model)
# logging.info(f"✅ Last Conv2D Layer: {last_conv_layer_name}")

# # Grad-CAM Function


# def grad_cam(model, img, class_index, conv_layer_name):
#     class_index = int(class_index)
#     grad_model = tf.keras.models.Model(
#         inputs=model.inputs,
#         outputs=[model.get_layer(conv_layer_name).output, model.outputs[0]]
#     )

#     # Convert image to TensorFlow tensor
#     img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    
#     if len(img_tensor.shape) == 3:  # If missing batch dimension, add it
#         img_tensor = tf.expand_dims(img_tensor, axis=0)

#     with tf.GradientTape() as tape:
#         tape.watch(img_tensor)  # Now, this is a TensorFlow tensor
#         conv_outputs, predictions = grad_model(img_tensor, training=False)
#         loss = predictions[:, class_index]

#     grads = tape.gradient(loss, conv_outputs)
#     if grads is None:
#         raise ValueError("❌ Gradient computation failed. Ensure the model's output is correct.")

#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0] * pooled_grads[None, None, :]

#     heatmap = np.mean(conv_outputs, axis=-1)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1

#     return heatmap


# # Overlay Heatmap on Image
# def overlay_grad_cam(image_path, heatmap, predicted_label):
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (224, 224))
    
#     heatmap = cv2.resize(heatmap, (224, 224))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     overlayed_image = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.title("Original Image")
#     plt.axis("off")
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(overlayed_image)
#     plt.title(predicted_label)
#     plt.axis("off")
    
#     plt.show()

# # Main Execution
# if __name__ == "__main__":
#     IMAGE_PATH = r'C:\Users\Anshu Kumar Rajak\Desktop\Skin\skin-disease-detection\dataset\acne\image1.png'
#     prediction = predict_skin_disease(model, IMAGE_PATH)
#     if prediction is None or prediction[0] == "Uncertain":
#         logging.warning("⚠️ Uncertain prediction. Grad-CAM not generated.")
#     else:
#         predicted_label, class_index = prediction
#         img_array = preprocess_image(IMAGE_PATH)
#         _ = model(img_array, training=False)
#         heatmap = grad_cam(model, img_array, class_index, last_conv_layer_name)
#         logging.info(f"Predicted Disease: {predicted_label}")
#         overlay_grad_cam(IMAGE_PATH, heatmap, predicted_label)
#         print(f"Predicted Disease: {predicted_label}")



import numpy as np
import tensorflow as tf
import cv2
import logging
import os
import uuid
from django.conf import settings
import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from detection_app.model_utils import load_trained_model, preprocess_image , predict_skin_disease

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class_labels = [
    "Acne","Melanoma", "Rosacea", "Eczema","Impetigo","Tinea","Perleche","Ringworm","Chickenpox","Basal Cell Carcinoma","Cellulitis","Pitted Keratolysis","Pityrosporum Folliculitis","Shingles","Others"
]

class_info = {
    "Acne": {
        "Cause": "Overproduction of oil, clogged pores, bacteria, and inflammation.",
        "Symptoms": "Pimples, blackheads, whiteheads, nodules, and cysts.",
        "Common Areas": "Face, back, chest, and shoulders.",
        "Risk Factors": "Hormonal changes, stress, diet, genetics.",
        "Treatment": "Topical creams, antibiotics, retinoids, laser therapy."
    },
    "Melanoma": {
        "Cause": "Uncontrolled growth of melanocytes (pigment-producing cells), often due to UV exposure.",
        "Symptoms": "Irregular moles with asymmetry, border changes, color variations, diameter >6mm.",
        "Common Areas": "Sun-exposed skin (face, arms, back).",
        "Risk Factors": "Fair skin, UV exposure, family history.",
        "Treatment": "Surgery, chemotherapy, radiation, immunotherapy."
    },
    "Rosacea": {
        "Cause": "Unknown; possibly immune response, genetics, or environmental triggers.",
        "Symptoms": "Facial redness, visible blood vessels, swollen red bumps.",
        "Common Areas": "Nose, cheeks, forehead, chin.",
        "Risk Factors": "Sun exposure, spicy food, alcohol, stress.",
        "Treatment": "Topical antibiotics, laser therapy, avoiding triggers."
    },
    "Eczema": {
        "Cause": "Overactive immune system reaction to irritants.",
        "Symptoms": "Dry, itchy, inflamed skin, blisters.",
        "Common Areas": "Hands, feet, elbows, behind the knees.",
        "Risk Factors": "Allergies, asthma, stress, irritants.",
        "Treatment": "Moisturizers, corticosteroids, antihistamines."
    },
    "Impetigo": {
        "Cause": "Bacterial infection (Staphylococcus or Streptococcus).",
        "Symptoms": "Red sores, honey-colored crusts.",
        "Common Areas": "Face (around nose and mouth), hands.",
        "Risk Factors": "Poor hygiene, warm climate, skin injuries.",
        "Treatment": "Antibiotics (topical or oral)."
    },
    "Tinea": {
        "Cause": "Fungal infection (dermatophytes).",
        "Symptoms": "Red, itchy, ring-like rashes.",
        "Common Areas": "Scalp (tinea capitis), body (tinea corporis), groin (tinea cruris).",
        "Risk Factors": "Humid environment, sweating, shared personal items.",
        "Treatment": "Antifungal creams, oral antifungals."
    },
    "Perleche": {
        "Cause": "Yeast or bacterial infection in mouth corners.",
        "Symptoms": "Cracks, redness, pain at mouth corners.",
        "Common Areas": "Lips.",
        "Risk Factors": "Dry lips, dentures, licking lips, vitamin deficiencies.",
        "Treatment": "Antifungal creams, lip balm, vitamin supplementation."
    },
    "Ringworm": {
        "Cause": "Fungal infection.",
        "Symptoms": "Circular, red, scaly patches with clear center.",
        "Common Areas": "Arms, legs, torso.",
        "Risk Factors": "Skin contact, shared objects, warm climate.",
        "Treatment": "Topical antifungals, oral antifungals."
    },
    "Chickenpox": {
        "Cause": "Varicella-zoster virus.",
        "Symptoms": "Itchy red blisters, fever, fatigue.",
        "Common Areas": "Whole body.",
        "Risk Factors": "Close contact with infected person, weakened immunity.",
        "Treatment": "Antihistamines, antiviral drugs for severe cases."
    },
    "Basal Cell Carcinoma": {
        "Cause": "Uncontrolled basal cell growth, often from UV exposure.",
        "Symptoms": "Pearly bump, open sores, red patches.",
        "Common Areas": "Face, neck, scalp.",
        "Risk Factors": "Sun exposure, fair skin, genetics.",
        "Treatment": "Surgery, radiation, topical medications."
    },
    "Cellulitis": {
        "Cause": "Bacterial infection (Streptococcus or Staphylococcus).",
        "Symptoms": "Red, swollen, warm skin, fever.",
        "Common Areas": "Legs, arms, face.",
        "Risk Factors": "Skin wounds, diabetes, weakened immune system.",
        "Treatment": "Oral or IV antibiotics."
    },
    "Pitted Keratolysis": {
        "Cause": "Bacterial infection from prolonged moisture.",
        "Symptoms": "Small pits in soles of feet, bad odor.",
        "Common Areas": "Feet.",
        "Risk Factors": "Sweaty feet, tight shoes.",
        "Treatment": "Antibacterial creams, foot hygiene."
    },
    "Pityrosporum Folliculitis": {
        "Cause": "Yeast infection of hair follicles.",
        "Symptoms": "Itchy, acne-like bumps on chest, back.",
        "Common Areas": "Upper back, chest, shoulders.",
        "Risk Factors": "Oily skin, heat, sweating.",
        "Treatment": "Antifungal medications, skincare adjustments."
    },
    "Shingles": {
        "Cause": "Reactivation of the chickenpox virus.",
        "Symptoms": "Painful rash with blisters, burning sensation.",
        "Common Areas": "One side of the body, often torso.",
        "Risk Factors": "Age, weakened immunity, stress.",
        "Treatment": "Antiviral drugs, pain relief medications."
    },
    "Other": {
        "Cause": "Unknown or not categorized.",
        "Symptoms": "Varies based on condition.",
        "Common Areas": "May affect any part of the body.",
        "Risk Factors": "Dependent on the specific condition.",
        "Treatment": "Consult a healthcare professional for diagnosis and treatment."
    }
}




# Load trained model
model = load_trained_model()
if model is None:
    raise RuntimeError("❌ Model loading failed. Cannot proceed!")

# Ensure model is initialized
_ = model.predict(np.zeros((1, 224, 224, 3)))  # Dummy prediction

# Unfreeze model layers to allow gradient computation
model.trainable = True
for layer in model.layers:
    layer.trainable = True

# Get the last Conv2D layer
def get_last_conv_layer(model):
    """Finds the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("❌ No Conv2D layer found in the model!")

last_conv_layer_name = get_last_conv_layer(model)
logging.info(f"✅ Last Conv2D Layer: {last_conv_layer_name}")

# Grad-CAM Function
def grad_cam(model, img, class_index, conv_layer_name):
    """Generate Grad-CAM heatmap for a given image and class index."""
    
    class_index = int(class_index)  # Ensure class_index is an integer

    # Define a model that outputs both the feature maps and predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.outputs[0]]
    )

    # Ensure img has the correct shape (1, 224, 224, 3)
    if len(img.shape) == 4:
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)  # Image already batched
    else:
        img_tensor = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)  # Expand batch dim

    logging.debug(f"Image tensor shape before Grad-CAM: {img_tensor.shape}")

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)  # Ensure TensorFlow tracks it
        conv_outputs, predictions = grad_model(img_tensor, training=False)  # Ensure training=False

        logging.debug(f"conv_outputs shape: {conv_outputs.shape}")
        logging.debug(f"predictions shape: {predictions.shape}")

        loss = predictions[:, class_index]  # Extract loss for the target class

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("❌ Gradient computation failed. Ensure the model's output is correct.")
    
    logging.debug(f"grads shape: {grads.shape}")

    # Compute pooled gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    logging.info(f"pooled_grads shape: {pooled_grads.shape}")

    # Apply weights to feature maps using broadcasting
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    conv_outputs = conv_outputs * pooled_grads[None, None, :]

    # Generate heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1  # Normalize (avoid division by zero)

    return heatmap

# Overlay Heatmap on Image
def overlay_grad_cam(image_path, heatmap, output_path):
    """Overlay Grad-CAM heatmap on the original image and save it."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlayed_image = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))

    # return output_path



def predict_and_generate_grad_cam(img_path, predict_skin_disease):
    try:
        logging.info(f"🔎 Processing image for prediction: {img_path}")

        # ✅ Check if file exists
        if not os.path.exists(img_path):
            logging.error("❌ Image file does not exist!")
            return {'disease_name': 'Error', 'causes': 'Image file missing'}

        # ✅ Preprocess Image
        img = preprocess_image(img_path)
        if img is None:
            logging.error("❌ Failed to preprocess image.")
            return {'disease_name': 'Error', 'causes': 'Error preprocessing image'}

        # ✅ Ensure model is loaded
        if model is None:
            logging.error("❌ Model is not loaded!")
            return {'disease_name': 'Error', 'causes': 'Model not available'}

        # ✅ Make Prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]

        if predicted_class >= len(class_labels):
            logging.error("⚠️ Predicted class index is out of range!")
            return {'disease_name': 'Unknown', 'causes': 'Invalid class index'}

        predicted_label = class_labels[predicted_class]
        causes = class_info.get(predicted_label, 'Unknown causes')
        logging.info(f"✅ Predicted Disease: {predicted_label}")

        # ✅ Generate Grad-CAM
        last_conv_layer_name = get_last_conv_layer(model)
        heatmap = grad_cam(model, img, predicted_class, last_conv_layer_name)

        # ✅ Save Grad-CAM Image
        grad_cam_filename = f"grad_cam_{uuid.uuid4().hex}.png"
        grad_cam_output_path = os.path.join(settings.MEDIA_ROOT, 'grad_cam', grad_cam_filename)
        overlay_grad_cam(img_path, heatmap, grad_cam_output_path)

        # ✅ Save Grad-CAM path
        predict_skin_disease.grad_cam_path = f"grad_cam/{grad_cam_filename}"
        predict_skin_disease.save()

        logging.info(f"✅ Grad-CAM saved at: {predict_skin_disease.grad_cam_path}")

        return {
            'disease_name': predicted_label,
            'causes': causes,
            'class_index': predicted_class
        }

    except Exception as e:
        logging.error(f"❌ Error in prediction: {e}")
        return {'disease_name': 'Error', 'causes': 'Prediction failed'}


