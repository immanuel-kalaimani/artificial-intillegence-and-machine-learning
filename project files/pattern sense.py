# Pattern Sense: Classy Ingredients Fabric Pattern Classification
# Full Executable Python Code

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# ===========================
# Step 1: Dataset Preparation
# ===========================
train_path = 'dataset/train'
val_path = 'dataset/validation'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(100, 100),
    batch_size=10,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_path,
    target_size=(100, 100),
    batch_size=10,
    class_mode='categorical'
)

# ===========================
# Step 2: Model Building
# ===========================
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===========================
# Step 3: Model Training
# ===========================
model.fit(train_data, epochs=5, validation_data=val_data)

# ===========================
# Step 4: Save the Model
# ===========================
model.save('fabric_pattern_model.h5')
print("Model training completed and saved successfully!")

# ===========================
# Step 5: Test New Image
# ===========================
def test_new_image(img_path):
    class_names = list(train_data.class_indices.keys())

    img = load_img(img_path, target_size=(100, 100))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    print("Predicted Fabric Pattern:", class_names[predicted_class])

# ===========================
# Example Test Image
# ===========================
# Replace 'test_image.jpg' with your test image path
test_new_image('test_image.jpg')