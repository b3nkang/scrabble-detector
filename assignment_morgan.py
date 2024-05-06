import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the path to the training and validation dataset
dataset_path = 'English/Fnt'
# dataset_path = '/Users/morganlo/deepLearning/Scrabblers/English/Fnt'

# Setup the ImageDataGenerator for preprocessing and data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,    
    validation_split=0.2,  
    rotation_range=10, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    zoom_range=0.1  
)

# Generate batches of tensor image data for training with real-time data augmentation
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',  
    subset='training'  
)

# Generate batches of tensor image data for validation
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation' 
)

# Define the input shape and model architecture
inputs = Input(shape=(128, 128, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(63, activation='softmax')(x)  

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    # steps_per_epoch=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=7
)

# Define the path to the testing dataset
test_dataset_path = 'split_cells/fs4_run_split'
image_files = [test_dataset_path + '/' + f for f in os.listdir(test_dataset_path) if f.endswith('.png')]
test_df = pd.DataFrame({
    'filename': image_files
})

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,  
    x_col='filename',
    y_col=None, 
    target_size=(128, 128),
    class_mode=None,  
    batch_size=32,
    shuffle=False
)

results = model.predict(test_generator)
test_images = next(test_generator)

predictions = model.predict(test_images)

classes = {i: str(i) for i in range(1, 11)}  # Digits 0-9 are classes 1-10
classes.update({i: chr(i + 54) for i in range(11, 37)})  # A-Z are classes 11-36
classes.update({i: chr(i + 60) for i in range(37, 63)})  # a-z are classes 37-62
classes[63] = 'Blank'  # Blank tile is class 63


rows, cols = (15,15)
# board = np.zeros((15, 15))
board = [
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_', '_']
]


for i in range(len(predictions)):
    max_pred_index = np.argmax(predictions[i]) + 1 
    max_pred_value = predictions[i][max_pred_index - 1]  
    character = classes[max_pred_index]
    row = i // 15
    col = i % 15
    board[row][col] = character
  
    print(f"Predictions for the {i+1} image: {character} with probability {max_pred_value:.2f}")

print(board)