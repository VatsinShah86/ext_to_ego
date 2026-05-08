import cv2
import numpy as np

# Define the board
SQUARES_X = 7        # number of chessboard squares horizontally
SQUARES_Y = 5        # number of chessboard squares vertically
SQUARE_SIZE = 0.0215   # meters — measure this on your printout with a ruler
MARKER_SIZE = 0.016   # meters — must be < SQUARE_SIZE

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_SIZE,
    MARKER_SIZE,
    dictionary
)

# Save as image to print
board_image = board.generateImage((2000, 1400))
cv2.imwrite("charuco_board.png", board_image)