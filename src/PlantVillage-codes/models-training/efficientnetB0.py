import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.applications import EfficientNetB0
import os

# ✅ Enable TensorFlow optimizations for CPU usage
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.optimizer.set_jit(True)

# ✅ Define dataset split directories
BASE_SPLIT_DIR = "data\PlantVillage-data"
TRAIN_DIR = os.path.join(BASE_SPLIT_DIR, "train")
VALID_DIR = os.path.join(BASE_SPLIT_DIR, "val")
TEST_DIR = os.path.join(BASE_SPLIT_DIR, "test")

# ✅ Image size and batch configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# ✅ Define data augmentation layer
data_augmentation = keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.3),  # Rotate up to 30% of a circle
    RandomZoom(0.1),  # Zoom in/out up to 10%
    RandomContrast(0.2),  # Adjust contrast by up to 20%
])

# ✅ Load datasets
raw_train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

raw_valid_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

raw_test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# ✅ Extract class names
class_names = raw_train_dataset.class_names
NUM_CLASSES = len(class_names)
print(f"🔢 Number of detected classes: {NUM_CLASSES}")
print(class_names)

# ✅ Apply augmentation to training data only
def augment_images(image, label):
    return data_augmentation(image, training=True), label

# Apply augmentation and prefetch
train_dataset = raw_train_dataset.map(augment_images, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
valid_dataset = raw_valid_dataset.prefetch(AUTOTUNE)
test_dataset = raw_test_dataset.prefetch(AUTOTUNE)

# ✅ Load Pre-trained EfficientNetB0 Model (Without Top Layers)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze lower layers (retain pre-trained features)
for layer in base_model.layers[:150]:
    layer.trainable = False

# Fine-tune higher layers
for layer in base_model.layers[150:]:
    layer.trainable = True

# Build Custom top layer
inputs = keras.Input(shape=(224, 224, 3))
x = inputs

# EfficientNetB0 expects inputs to be preprocessed in a specific way
# We'll use the built-in preprocessing function
x = tf.keras.applications.efficientnet.preprocess_input(x)

# Connect to base model
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

# ✅ Define the Model
model = Model(inputs, outputs)

# ✅ Compile the Model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Add Early Stopping & Model Checkpointing
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,  # Increased patience since augmentation may slow convergence
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    "efficientnetB0_model_augmented.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# ✅ Train the Model with Augmentation
EPOCHS = 30  
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]
)

# ✅ Load the best model (lowest val_loss)
best_model = keras.models.load_model("efficientnetB0_model_augmented.keras")

# ✅ Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(test_dataset)
print(f"🏆 Final Test Accuracy (Best Model): {test_acc * 100:.2f}%")