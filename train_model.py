import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# -----------------------------
# PATHS
# -----------------------------
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5   # 🔥 STEP 3: Reduced epochs for safe fine-tuning

# -----------------------------
# DATA GENERATORS
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# -----------------------------
# LOAD PRETRAINED MODEL
# -----------------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# 🔥 STEP 1: Fine-tuning (VERY IMPORTANT)
# Freeze most layers
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Unfreeze last 10 layers
for layer in base_model.layers[-10:]:
    layer.trainable = True

# -----------------------------
# CUSTOM CLASSIFICATION HEAD
# -----------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=0.00001),  # 🔥 STEP 2: Lower LR for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("model", exist_ok=True)
model.save("model/pneumonia_model.h5")

print("✅ Fine-tuned model trained and saved successfully.")
