# py-camloc

locator using two cameras???

## Calibration

Camera calibration works the same way as described in the [official opencv documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).

1. edit `config.py` to match your camera setup (resolution, chessboard tiles, number of samples, delay)
2. find a chessboard pattern (I used [this](https://raw.githubusercontent.com/MarkHedleyJones/markhedleyjones.github.io/master/media/calibration-checkerboard-collection/Checkerboard-A1-90mm-8x5.svg))
3. hold the chessboard perpendicularly to the camera, move it to different sections and edges of the screen for better results
4. calibration results will be saved to `.npy` numpy binary files
