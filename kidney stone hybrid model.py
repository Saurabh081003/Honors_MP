import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121, VGG16
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Concatenate, Dropout, 
    BatchNormalization, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import shutil
import matplotlib.pyplot as plt

# === SET PATH ===
dataset_path = "/kaggle/input/ct-kidney-dataset-normal-cyst-tumor-and-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
output_dir = "/kaggle/working/split_dataset"
os.makedirs(output_dir, exist_ok=True)

# === CREATING TRAIN, VALIDATION, TEST SPLIT ===
all_images = []
all_labels = []

for class_name in os.listdir(dataset_path):  
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            all_images.append(img_path)
            all_labels.append(class_name)

df = pd.DataFrame({"image": all_images, "label": all_labels})

# Split dataset: 70% Train, 15% Val, 15% Test
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

# Create directories for train, val, test
for split in ['train', 'val', 'test']:
    for class_name in df['label'].unique():
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

# Function to copy images to their respective directories
def copy_images(df, split_name):
    for idx, row in df.iterrows():
        src_path = row['image']
        dest_dir = os.path.join(output_dir, split_name, row['label'])
        shutil.copy2(src_path, dest_dir)

# Copy images to their respective directories
copy_images(train_df, 'train')
copy_images(val_df, 'val')
copy_images(test_df, 'test')

print("Dataset splitting and copying completed!")
print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 4  # tumor, cyst, stone, normal

# Define class names
CLASS_NAMES = sorted(df["label"].unique())

# === DATA GENERATORS ===
# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Validation and test data (only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Calculate class weights for imbalanced data
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

# === HYBRID MODEL ARCHITECTURE ===
def create_hybrid_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Create separate models with unique names
    # ResNet50
    resnet = ResNet50(
        weights="imagenet", 
        include_top=False, 
        input_tensor=Input(shape=input_shape),
        name='resnet'
    )
    # DenseNet121
    densenet = DenseNet121(
        weights="imagenet", 
        include_top=False, 
        input_tensor=Input(shape=input_shape),
        name='densenet'
    )
    # VGG16
    vgg = VGG16(
        weights="imagenet", 
        include_top=False, 
        input_tensor=Input(shape=input_shape),
        name='vgg'
    )
    
    # Freeze base models
    for model in [resnet, densenet, vgg]:
        for layer in model.layers:
            layer.trainable = False
    
    # Get outputs from each model
    resnet_output = resnet(input_layer)
    densenet_output = densenet(input_layer)
    vgg_output = vgg(input_layer)
    
    # Extract features from each model
    resnet_features = GlobalAveragePooling2D(name='resnet_gap')(resnet_output)
    densenet_features = GlobalAveragePooling2D(name='densenet_gap')(densenet_output)
    vgg_features = GlobalAveragePooling2D(name='vgg_gap')(vgg_output)
    
    # Combine features
    combined = Concatenate(name='combined_features')([resnet_features, densenet_features, vgg_features])
    
    # Custom classification head
    x = Dense(1024, activation='relu', name='fc1')(combined)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.5, name='dropout1')(x)
    
    x = Dense(512, activation='relu', name='fc2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Dropout(0.4, name='dropout2')(x)
    
    x = Dense(256, activation='relu', name='fc3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Dropout(0.3, name='dropout3')(x)
    
    x = Dense(128, activation='relu', name='fc4')(x)
    x = Dropout(0.2, name='dropout4')(x)
    
    # Output layer
    output_layer = Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create model
model = create_hybrid_model()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

# Print model summary
model.summary()

# === CALLBACKS ===
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_hybrid_model.h5', monitor='val_accuracy', save_best_only=True)
]

# === TRAINING ===
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# === EVALUATION ===
# Load best model
model.load_weights('best_hybrid_model.h5')

# Evaluate on test set
test_results = model.evaluate(test_generator)
print(f"Test Loss: {test_results[0]}")
print(f"Test Accuracy: {test_results[1]}")
print(f"Test Precision: {test_results[2]}")
print(f"Test Recall: {test_results[3]}")
print(f"Test AUC: {test_results[4]}")

# === PLOTTING ===
# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# === PREDICTION FUNCTION ===
def predict_image(image_path):
    """Predict class for a single image"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence

# Example usage:
# predicted_class, confidence = predict_image('path/to/your/image.jpg')
# print(f"Predicted: {predicted_class} with {confidence:.2%} confidence")