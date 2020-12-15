import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure


class Window(tk.Frame):
    """ Creates the GUI """
    def __init__(self, parent, model, *args, **kwargs):
        """
        Initializes the window.

        Args:
            parent (tk.Tk): Tk root.
        """
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.model = model
        self.parent.title("Digit Recognizer")
        self.parent.resizable(False, False)

        # Canvas
        self.number_canvas = DrawCanvas(self, width=200, height=200, bg='white')
        self.number_canvas.grid(row=1, column=0, padx=(5, 0))

        self.input_canvas = DisplayCanvas(self, width=200, height=200, bg='white')
        self.input_canvas.grid(row=1, column=1, padx=(0, 5))

        self.barplot_canvas = DisplayCanvas(self, width=406, height=200, bg='white')
        self.barplot_canvas.grid(row=2, columnspan=2, pady=(0, 5))

        self.clear_button = tk.Button(self, text='clear', command=self.clear)
        self.clear_button.grid(row=0, column=0)

        self.predict_button = tk.Button(self, text='predict', command=self.predict)
        self.predict_button.grid(row=0, column=1)
        self.pack()

    def clear(self):
        """ Removes the contents from the drawing, the input and the plot canvas. """
        self.number_canvas.delete('all')
        self.input_canvas.delete('all')
        self.barplot_canvas.delete('all')

    def predict(self):
        """ Inputs the image to the network for predicting the number. """
        self.number_canvas.postscript(file='prediction/result.eps')
        img_gray = Image.open('prediction/result.eps').convert('L')
        img_res = img_gray.resize((28, 28))
        img = np.invert(np.asarray(img_res)) / 255.0
        self.show_img(img)
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        score = self.model.predict(img)
        self.show_barplot(score[0])

    def show_img(self, img):
        """
        Displays the input image.

        Args:
            img (np.array): Input array rescaled to 28x28.
        """
        fig = Figure(figsize=(3, 3))
        ax = fig.gca()
        ax.imshow(img)
        ax.axis('off')
        fig.savefig('prediction/result.png', bbox_inches='tight', pad_inches=0)
        self.input_canvas.display('prediction/result.png')

    def show_barplot(self, score):
        """
        Displays the barplot of scores.

        Args:
            score (list): Probability for each of the numbers.
        """
        fig = Figure(figsize=(4, 2))
        ax = fig.gca()
        ax.bar(range(len(score)), score)
        ax.set_xticks(range(len(score)))
        ax.set_ylim([0, 1])
        fig.savefig('prediction/dist.png')
        self.barplot_canvas.display('prediction/dist.png')


class DrawCanvas(tk.Canvas):
    """ Drawing window. """
    r = 10  # Stroke size
    color = 'black'

    def __init__(self, parent, *args, **kwargs):
        """
        Initializes the drawing canvas.

        Args:
            parent (tk.Frame): Main GUI.
        """
        tk.Canvas.__init__(self, parent, *args, **kwargs)
        self.bind('<B1-Motion>', self.draw)

    def draw(self, event):
        """
        Draws a circle on click event.

        Args:
            event (tk.Event): Click position.
        """
        color = 'black'
        self.create_oval(event.x - self.r, event.y - self.r, event.x + self.r, event.y + self.r,
                         fill=color, outline=color)


class DisplayCanvas(tk.Canvas):
    """ Network input window. """
    def __init__(self, parent, *args, **kwargs):
        """
        Initializes the display canvas.

        Args:
            parent (tk.Frame): Main GUI.
        """
        tk.Canvas.__init__(self, parent, *args, **kwargs)
        self.image_canvas = None

    def display(self, img):
        """
        Plots the image in the canvas.

        Args:
            img (np.array): Input array rescaled to 28x28.
        """
        self.image_canvas = ImageTk.PhotoImage(Image.open(img))
        self.create_image(0, 0, anchor='nw', image=self.image_canvas)
