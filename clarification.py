import cv2
import numpy as np


def denoise_erode(img, size, iterations, debug=0):
    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    binary = cv2.bitwise_not(binary)
    kernel = np.ones((size, size), np.uint8)
    denoise = cv2.erode(binary, kernel, iterations=iterations)
    denoise = cv2.bitwise_not(denoise)

    if debug:
        cv2.imshow('denoise', denoise)
        cv2.waitKey()
    return denoise


def denoise_open(img, size, iterations, debug=0):
    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    # define the kernel
    kernel = np.ones((size, size), np.uint8)

    # opening the image
    opening = cv2.morphologyEx(cv2.bitwise_not(binary), cv2.MORPH_OPEN, kernel, iterations=iterations)
    opening = cv2.bitwise_not(opening)

    if debug:
        cv2.imshow('opening', opening)
        cv2.waitKey()
    return opening


def contrast(img, blur_size, threshold):
    img = cv2.blur(img, (blur_size, blur_size))
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return img
