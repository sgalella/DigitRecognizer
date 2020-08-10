import tensorflow as tf

# Load the dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_valid = X_valid.reshape(10000, 28, 28, 1)

# Create convolutional network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create image generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,
                                                                rotation_range=20,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.1,
                                                                fill_mode='nearest')

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow(X_train, y_train)
validation_generator = validation_datagen.flow(X_valid, y_valid)

# Train model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

model.save('model/model.h5')
