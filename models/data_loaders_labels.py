from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

train_directory = '/Users/user/PycharmProjects/tank_streamlit/data/dataset/train'
validation_directory = '/Users/user/PycharmProjects/tank_streamlit/data/dataset/validation'
test_directory = '/Users/user/PycharmProjects/tank_streamlit/data/dataset/test'

num_classes = 10  # Update this to reflect the actual number of vehicle types

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(150, 150, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Setup the data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_validation_datagen.flow_from_directory(
    validation_directory,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

# # Train the model
model.fit(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=800 // batch_size
)

# After training the model, save it
# model.save('military_vehicle_model.h5')
# model.save_weights('military_vehicle_model_weights.h5')

# # Load the model
# from keras.models import load_model
# model = load_model('military_vehicle_model.h5')
# model.load_weights('military_vehicle_model_weights.h5')
# model.summary()

