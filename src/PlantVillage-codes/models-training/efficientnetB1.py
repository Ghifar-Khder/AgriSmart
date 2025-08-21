import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB1  # Use EfficientNetB1
import os

# ✅ Enable TensorFlow optimizations for CPU usage
tf.config.threading.set_inter_op_parallelism_threads(8)  # Adjust based on CPU cores
tf.config.threading.set_intra_op_parallelism_threads(8)  # Adjust based on CPU cores
tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra) for speed-up

# ✅ Define dataset split directories
BASE_SPLIT_DIR = "data\PlantVillage-data"
TRAIN_DIR = os.path.join(BASE_SPLIT_DIR, "train")
VALID_DIR = os.path.join(BASE_SPLIT_DIR, "val")
TEST_DIR = os.path.join(BASE_SPLIT_DIR, "test")

# ✅ Adjusted for EfficientNetB1
IMG_SIZE = (240, 240)  # Model input size
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE  # Optimizes data pipeline

# ✅ Load dataset using `image_dataset_from_directory()` (MUCH FASTER)
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

# ✅ Extract class names BEFORE applying `.prefetch()`
class_names = raw_train_dataset.class_names  # Extract class names first
NUM_CLASSES = len(class_names) # number of classes
print(f"🔢 Number of detected classes: {NUM_CLASSES}")
print(class_names)

# ✅ Apply Data Augmentation (Only to Training Data)
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),  # Flip images
    keras.layers.RandomRotation(0.2),  # Rotate images by up to 20%
    keras.layers.RandomZoom(0.2),  # Zoom images by up to 20%
    keras.layers.RandomTranslation(0.1, 0.1),  # Shift images in width & height
    keras.layers.RandomContrast(0.1)  # Adjust image contrast
])

# ✅ Apply Data Augmentation to Training Dataset
train_dataset = (
    raw_train_dataset
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)  # Apply augmentation
    .prefetch(buffer_size=AUTOTUNE)  # Optimize loading
)

valid_dataset = raw_valid_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = raw_test_dataset.prefetch(buffer_size=AUTOTUNE)

# ✅ Load Pre-trained EfficientNetB1 Model (Without Top Layers)
base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(240, 240, 3))

# **Freeze lower layers** (retain pre-trained features)
for layer in base_model.layers[:300]:  # Adjust number of layers to freeze
    layer.trainable = False

# **Fine-tune higher layers**
for layer in base_model.layers[300:]:
    layer.trainable = True

# **Build Custom Model**
x = GlobalAveragePooling2D()(base_model.output)  # Pooling
x = Dense(256, activation='relu')(x) 
x = Dropout(0.5)(x)  # Dropout to prevent overfitting
output = Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer

# ✅ Define the Model
model = Model(inputs=base_model.input, outputs=output)

# ✅ Compile the Model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ **Add Early Stopping & Model Checkpointing**
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=2,  # Stop if val_loss doesn't improve for 2 epochs
    restore_best_weights=True,  # Restore the best weights
    verbose=1
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    "efficientnetB1.keras",
    monitor='val_loss',  # Monitor validation loss
    save_best_only=True,  # Save only if val_loss improves
    verbose=1
)

# ✅ Train the Model (🚀 Faster Data Loading with `image_dataset_from_directory()` + Augmentation)
EPOCHS = 20  # Allow early stopping to determine the final number of epochs
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]  # Apply early stopping & save best model
)

# ✅ Load the best model (lowest val_loss)
best_model = keras.models.load_model("efficientnetB1.keras")  # Load model

# ✅ Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(test_dataset)
print(f"🏆 Final Test Accuracy (Best Model): {test_acc * 100:.2f}%")
