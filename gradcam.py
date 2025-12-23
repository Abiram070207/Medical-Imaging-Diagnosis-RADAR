import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "model/pneumonia_model.h5"
model = load_model(MODEL_PATH)

# -----------------------------
# IMAGE PREPROCESSING FUNCTION
# -----------------------------
def preprocess_image(img_path, img_size=224):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# -----------------------------
# GRAD-CAM FUNCTION
# -----------------------------
def generate_gradcam(img_path, last_conv_layer_name="conv5_block3_out"):
    img_array = preprocess_image(img_path)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

# -----------------------------
# OVERLAY HEATMAP ON IMAGE
# -----------------------------
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    return superimposed_img

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    image_path = "sample_xray.jpg"  # change this path

    heatmap = generate_gradcam(image_path)
    cam_image = overlay_heatmap(image_path, heatmap)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Original X-ray")
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()
