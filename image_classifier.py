import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1️⃣ Load dataset (CIFAR-10 has 10 image classes)
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values (0–1 range)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2️⃣ Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# 3️⃣ Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4️⃣ Train model
history = model.fit(x_train, y_train, epochs=5, 
                    validation_data=(x_test, y_test))

# 5️⃣ Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")

# 6️⃣ Save model
model.save('image_classifier_model.h5')
print("Model saved successfully!")

import numpy as np
import matplotlib.pyplot as plt

# Class names in CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Pick one random image
img = x_test[15]
plt.imshow(img)
plt.show()

# Predict
pred = model.predict(img.reshape(1, 32, 32, 3))
print("Predicted:", class_names[np.argmax(pred)])
