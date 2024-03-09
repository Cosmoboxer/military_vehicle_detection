from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.models import Model

def build_model(input_shape, num_classes):
    # Input layer
    input_layer = Input(shape=input_shape)

    # Feature extraction layers
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the feature maps
    x = Flatten()(x)

    # Fully connected layers for classification
    classification_output = Dense(512, activation='relu')(x)
    classification_output = Dropout(0.5)(classification_output)
    classification_output = Dense(num_classes, activation='softmax', name='classification_output')(classification_output)

    # Fully connected layers for bounding box regression
    bbox_output = Dense(512, activation='relu')(x)
    bbox_output = Dropout(0.5)(bbox_output)
    bbox_output = Dense(4, activation='linear', name='bbox_output')(bbox_output)  # 4 outputs for (x, y, w, h)

    # Define the model with two outputs
    model = Model(inputs=input_layer, outputs=[classification_output, bbox_output])

    # Compile the model
    model.compile(optimizer='adam',
                  loss={'classification_output': 'categorical_crossentropy',
                        'bbox_output': 'mse'},
                  metrics={'classification_output': 'accuracy',
                           'bbox_output': 'mse'})

    return model

# Specify the input shape and number of classes
input_shape = (150, 150, 3)  # Example input shape
num_classes = 4  # Example number of classes

# Build the model
model = build_model(input_shape, num_classes)

# Setup the data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_directory = '/Users/user/PycharmProjects/tank_streamlit/data/dataset/train'
validation_directory = '/Users/user/PycharmProjects/tank_streamlit/data/dataset/validation'



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

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=800 // batch_size
)

# Save the model
model.save('military_vehicle_coordinates_model.h5')
model.save_weights('military_vehicle_coordinates_model_weights.h5')
# Path: military_vehicle_detection/models/data_loaders_labels.py




