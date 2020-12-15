import tkinter as tk

import tensorflow as tf

from .window import Window

root = tk.Tk()
model = tf.keras.models.load_model('digit_recognizer/model/model.h5')
window = Window(root, model, width=420, height=450)
root.mainloop()
