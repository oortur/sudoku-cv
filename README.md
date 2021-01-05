# sudoku-cv
Image based sudoku solver

## Usage
To get solutions for your sudoku you need a photo of any sudoku and Python with installed Numpy and opencv. 
File `model.py` is used to get digit classifier for solving sudoku, `mnist.model` serves model weights.

To get solution run
```bash
python sudoku.py -i INPUT_IMAGE_PATH -o OUTPUT_IMAGE_PATH
```

## Examples

Find examples in `/images`:

<img src="https://github.com/balan/sudoku-cv/blob/main/images/image_1.jpg?raw=true" height=300 hspace=30> <img
src="https://github.com/balan/sudoku-cv/blob/main/images/solved_1.jpg?raw=true" height=300>

<img src="https://github.com/balan/sudoku-cv/blob/main/images/image_0.jpg?raw=true" height=200 hspace=30> <img 
src="https://github.com/balan/sudoku-cv/blob/main/images/solved_0.jpg?raw=true" height=200>
