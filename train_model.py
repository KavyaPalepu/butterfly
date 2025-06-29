import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# === Dataset path
BASE_PATH = "archive"

# === Load CSV and clean columns
train_df = pd.read_csv(os.path.join(BASE_PATH, "Training_set.csv"))
train_df.rename(columns={"Filename": "filename", "Class": "label"}, inplace=True)
train_df['filepath'] = train_df['filename'].apply(lambda x: os.path.join(BASE_PATH, "train", x))

# === Optional: Use a subset for faster testing (remove this for full training)
# train_df = train_df.sample(frac=0.3, random_state=42)

# === Image and training config
img_size = (160, 160)
batch_size = 32
num_classes = train_df['label'].nunique()

# === Data generators
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

# === Load MobileNetV2 base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
base_model.trainable = False  # Freeze base layers

# === Build classifier
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

# === Train the model (with multiprocessing)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[checkpoint, early_stop]
)

