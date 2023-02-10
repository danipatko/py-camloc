#####################
#   CALIBRATE.PY    #
#####################

# numpy bin file prefix
SAVE_TO = "sjcam/sjcam"

# number of samples to take (the more the better)
TEST_COUNT = 20

# number of inner columns on the chessboard
INNER_ROW_LENGTH = 8

# number of inner rows on the chessboard
INNER_COL_LENGTH = 5

# seconds before taking picture
VID_DELAY = 3

# seconds to show result image for
RES_DELAY = 1

# camera resolution
WIDTH = 1280
HEIGHT = 720

# create newcameramatrix object for rescaling and transforming
RESCALE = None  # (1920, 1080)

#####################
#    TRESHOLD.PY    #
#####################

# adjust accoording to lights (the higher, the less white areas)
BIN_TRESHOLD = 0.8 * 255

# minimal rectangle area
MIN_AREA = 20

# maximal rectangle area
MAX_AREA = 1000

# maximal difference of the rectangle's side ratio
SQUARE_RATIO_OFFSET = 0.1

# drop a frame if the average and the detected point is too far away from each other
DIST_IGNORE = 100

# first elem is the latest frame
WEIGHTS = [5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
