import warnings

import cv2
import numpy as np
from scipy.signal import find_peaks

def contrast(img, blur_size, threshold):
    img = cv2.blur(img, (blur_size, blur_size))
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return img


def find_rectangle_contain_all_contours(contours):
        x1, y1, w1, h1 = cv2.boundingRect(contours[0])
        x2 = x1 + w1
        y2 = y1 + h1
        for cnt in contours:
            _x1, _y1, _w1, _h1 = cv2.boundingRect(cnt)
            _x2 = _x1 + _w1
            _y2 = _y1 + _h1
            x1 = min(x1, _x1)
            y1 = min(y1, _y1)
            x2 = max(x2, _x2)
            y2 = max(y2, _y2)
        return x1, y1, x2, y2

def denoise_open(img, size, iterations, debug = 0):
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


def floor_to_list(num, floor_list):
    """
    Returns the greatest number in `floor_list` that is <= num.
    If no such number exists, returns None.
    """
    if floor_list is not None:
        candidates = [n for n in floor_list if n <= num]
        return max(candidates, default=0)
    else:
        return None

def element_wise_sum(*lists):
    """ This will return a list that each of its elements is the sum of elements in list_a and list_b """
    return [sum(p) for p in zip(*lists)]


def extract_image_at_region(image, region, safety_margin = (0, 0, 0, 0)):
    """
    Extract a region from an image with a safety margin.
    :param image: Input image.
    :param region: Desired region to extract from the image in the format of (x1, y1, x2, y2).
    :param safety_margin: Safety margin to add to the desired region. Format: (left, up, right, down)
    :return: Extracted image from the region plus safety margin and the rectangle of safe region in the format of (x1, y1, x2, y2).
    """
    # Region format: x1, y1, x2, y2
    # NumPy image format: y1, y2, x1, x2
    desired_region_x1 = max(region[0] - safety_margin[0], 0)
    desired_region_y1 = max(region[1] - safety_margin[1], 0)
    desired_region_x2 = min(region[2] + safety_margin[2], image.shape[1])
    desired_region_y2 = min(region[3] + safety_margin[3], image.shape[0])
    safe_region_rect = (desired_region_x1, desired_region_y1, desired_region_x2, desired_region_y2)
    safe_region_image = image[desired_region_y1:desired_region_y2, desired_region_x1:desired_region_x2]
    return safe_region_image, safe_region_rect


def add_colored_border(image, border_width=2, top_color=(0, 255, 0), bottom_color=(0, 255, 0),
                       left_color=(0, 255, 0), right_color=(0, 255, 0)):
    """
    Add a border around a cv2 image with each side in a specified color.

    Args:
        image: Input cv2 image (numpy array in BGR format)
        border_width: Width of the border in pixels (default: 4)
        top_color: BGR tuple for top border color (default: green)
        bottom_color: BGR tuple for bottom border color (default: red)
        left_color: BGR tuple for left border color (default: blue)
        right_color: BGR tuple for right border color (default: yellow)

    Returns:
        Output cv2 image with colored borders added
    """
    if image is None or image.size==0:
        # Create an empty cv2 image 70x140 with "Empty" text on it:
        image = np.zeros((70, 140, 3), dtype=np.uint8)
        cv2.putText(image, "None!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Validate color tuples
    for color, name in [(top_color, "top_color"), (bottom_color, "bottom_color"),
                        (left_color, "left_color"), (right_color, "right_color")]:
        if not (isinstance(color, tuple) and len(color) == 3 and all(0 <= c <= 255 for c in color)):
            raise ValueError(f"{name} must be a BGR tuple with values between 0 and 255")

    # Get image dimensions
    height, width = image.shape[:2]

    # Create a new image with space for borders
    new_height = height + 2 * border_width
    new_width = width + 2 * border_width
    bordered_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Copy the original image to the center
    bordered_image[border_width:border_width + height,
    border_width:border_width + width] = image

    # Draw top border
    bordered_image[0:border_width, :] = top_color

    # Draw bottom border
    bordered_image[height + border_width:, :] = bottom_color

    # Draw left border (including corners to avoid overlap)
    bordered_image[:, 0:border_width] = left_color

    # Draw right border (including corners to avoid overlap)
    bordered_image[:, width + border_width:] = right_color

    return bordered_image


def cv2_imshow_with_border(window_name, image):
    cv2.imshow(window_name, add_colored_border(image))


def find_containing_region(img, debug=False):
    img_contrast = contrast(img, 5, 210)

    img_contours, _ = cv2.findContours(cv2.bitwise_not(img_contrast), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if img_contours == ():
        return None

    if debug:
        img_debug = img.copy()
        cv2.drawContours(img_debug, img_contours, -1, (0, 100, 100), 30)
        cv2_imshow_with_border(img_debug)

    x1,y1,x2,y2 = find_rectangle_contain_all_contours(img_contours)
    return x1, y1, x2, y2

def remove_empty_space_around_image(image, safety_margin = (0, 0, 0, 0), sides = (1, 1, 1, 1)):
    """
    :param image: Input image.
    :param safety_margin: Add safety white band to the output image.
    :param sides: Remove white space on the selected sides (left, up, right, down). Select = 1, Not selected = 0.
    :return: Automatically white space on the selected sides will be cropped.
    """
    # Region format: x1, y1, x2, y2
    # NumPy image format: y1, y2, x1, x2

    x1, y1, x2, y2 = find_containing_region(image)

    x1 = max(x1 - safety_margin[0], 0)
    y1 = max(y1 - safety_margin[1], 0)
    x2 = min(x2 + safety_margin[2], image.shape[1])
    y2 = min(y2 + safety_margin[3], image.shape[0])

    if sides[0]==0: x1 = 0
    if sides[1]==0: y1 = 0
    if sides[2]==0: x2 = image.shape[1]
    if sides[3]==0: y2 = image.shape[0]

    return image[y1:y2, x1:x2]


def resize_image_by_new_shap(image, new_shape):
    """
    Resize an input image to the specified new shape using OpenCV.

    Args:
        image: Input image (numpy array) to be resized
        new_shape: Tuple of (width, height) for the new image size

    Returns:
        Resized image as a numpy array
    """

    # Get the new width and height
    new_width, new_height = new_shape

    # Resize the image using OpenCV's resize function
    # INTER_LINEAR is used for both upscaling and downscaling as a good balance
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image

def resize_image_by_ratio(image, ratio):
    """
    Resize an input image by a specified ratio using OpenCV.

    Args:
        image: Input image (numpy array) to be resized
        ratio: Ratio to resize the image (e.g., 0.5 for 50% of the original size)

    Returns:
        Resized image as a numpy array
    """

    # Get the original width and height
    original_height, original_width = image.shape[:2]

    # Calculate the new width and height based on the ratio
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize the image using OpenCV's resize function
    # INTER_LINEAR is used for both upscaling and downscaling as a good balance

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image


def draw_dividing_lines(image_column, dividing_lines, entries_y, head_words_rect):

    w, h = image_column.shape[1], image_column.shape[0]
    output_img = cv2.cvtColor(image_column, cv2.COLOR_GRAY2BGR)
    # Draw horizontal dividing lines between text lines
    for y in dividing_lines:
        cv2.line(output_img, (0, y), (w, y), (255, 0, 255), 1)

    # Draw horizontal entries dividing lines where an entry detected
    for y in entries_y:
        cv2.line(output_img, (0, y), (w, y), (255, 0, 0), 2)

    for x1,y1,x2,y2 in head_words_rect:
        cv2.rectangle(output_img, (x1,y1), (x2, y2), (0, 0, 255), 1)

    # Draw vertical entry detection line
    cv2.line(output_img, (w - 40, 0), (w - 40, h), (0, 0, 255), 1)

    cv2.line(output_img, (10, 0), (10, h), (0, 0, 255), 1)

    return output_img

def remove_small_areas(image, min_area):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(image, [cnt], -1, (255, 255, 255), -1)
    return image


def is_rectangle_nonempty(rectangle, image):
    x1, y1, x2, y2 = rectangle
    return np.any(image[y1:y2, x1:x2] == 0)