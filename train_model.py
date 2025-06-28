import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# === Dataset path
BASE_PATH = "archive"

# === Load the CSV and rename columns
train_df = pd.read_csv(os.path.join(BASE_PATH, "Training_set.csv"))
train_df.rename(columns={"Filename": "filename", "Class": "label"}, inplace=True)

# === Add full image paths
train_df['filepath'] = train_df['filename'].apply(lambda x: os.path.join(BASE_PATH, "train", x))

# === Model parameters
img_size = (224, 224)
batch_size = 32
num_classes = train_df['label'].nunique()

# === Image data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='label',
    target_size=img_size,
    class_mode='categorical',
    subset='training',
    batch_size=batch_size,
    shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='label',
    target_size=img_size,
    class_mode='categorical',
    subset='validation',
    batch_size=batch_size,
    shuffle=True
)

# === Load VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# === Add custom classifier on top
model = Sequential([
    base_model,
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# === Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Callbacks
checkpoint = ModelCheckpoint("vgg16_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=5)

# === Train the model
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[checkpoint, early_stop]
)
