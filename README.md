# Digit Recognizer
Interface to visualize CNN predictions of digits drawn by the user.

<p align="center">
    <img width="400" height="400"src="images/digit_recognizer.gif">
</p>
The top-left square is the canvas, in which a number between 0 and 9 is expected. The top-right shows the input to the CNN: the drawing has been reduced to have a size 28x28. The bottom plot shows the probabilities predicted by the CNN.



## Installation
The code was tested using Python 3.6.10. To install the necessary packages run:

```bash
pip install -r requirements.txt
```

If using Conda, you can also create an environment with the requirements:

```bash
conda env create -f environment.yml
```

By default the environment name is `digit-recognizer`. To activate it run:

```bash
conda activate digit-recognizer
```


## Usage

To run the GUI:

```python
python -m digit_recognizer
```

`digit_recognizer/model/` contains a trained CNN. You change the architecture or the parameters in `digit_recognizer/cnn_model.py`. To train it run:

```python
python digit_recognizer/cnn_model.py
```
`prediction/` stores the input and the scores for the last prediction.
