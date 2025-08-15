import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the data
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One-hot encode labels
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build the model
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')

# Save the model
model.save("mnist.h5")

# GUI for digit drawing
class DigitRecognizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack()
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()
        self.button_predict = tk.Button(self, text="Recognise", command=self.predict_digit)
        self.button_predict.pack()
        self.canvas.bind('<B1-Motion>', self.paint)
        self.image = Image.new('L', (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result = tk.Label(self, text="", font=("Helvetica", 48))
        self.result.pack()

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+10, y+10, fill='black', width=10)
        self.draw.ellipse([x, y, x+10, y+10], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result.config(text="")

    def preprocess_image(self):
        image = self.image.resize((28, 28)).convert('L')
        image = np.array(image)
        image = 255 - image  # Invert colors
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        return image

    def predict_digit(self):
        processed_image = self.preprocess_image()
        prediction = model.predict(processed_image)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        self.result.config(text=f"{digit}, {confidence:.2f}%")

app = DigitRecognizer()
app.mainloop()
