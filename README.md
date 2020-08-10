# Digit Recognizer
Interface to visualize CNN predictions of digits drawn by the user.

<p align="center">
    <img width="400" height="400"src="images/digit_recognizer.gif">
</p>



## Installation

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```



## Usage

To run the GUI:

```python
python digit_recognizer.py
```

`model/` contains a trained CNN. You change the architecture or the parameters in `cnn_model.py`. To train it run:

```python
python cnn_model.py
```

