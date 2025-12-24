from auxiliaries import *
from clarification import *

import cv2
import numpy as np

import warnings


def find_heading_line_position(page_image, heading_line_detection_height_up=50, heading_line_detection_height_down=250, debug=0):
    """

    :param page_image:
    :param heading_line_detection_height_up:
    :param heading_line_detection_height_down:
    :param debug:
    :return:
    """
    initial_head_cut = page_image[
                       heading_line_detection_height_up:heading_line_detection_height_down]
    blur = cv2.blur(initial_head_cut, (5, 5))
    _, binary = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLines(cv2.bitwise_not(binary), 1, np.pi / 1800, 60, min_theta=np.pi / 2 - 0.01,
                           max_theta=np.pi / 2 + 0.01)

    if lines is None:
        print("No heading line detected. Check heading line detection configs.")
        return None  # No heading line was detected.

    heading_line_y = int(np.average([line[0] for line in lines[0]])) + heading_line_detection_height_up

    head_line_cut_up = max(0, heading_line_y - 15)
    head_line_cut_down = min(heading_line_y + 15, page_image.shape[0])
    head_line_cut = page_image[head_line_cut_up:head_line_cut_down]
    head_line_cut = cv2.blur(head_line_cut, (5, 5))
    _, head_line_cut = cv2.threshold(head_line_cut, 210, 255, cv2.THRESH_BINARY)

    heading_line_x1 = 0
    heading_line_x2 = 0
    contours, _ = cv2.findContours(cv2.bitwise_not(head_line_cut), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(head_line_cut, contours, -1, (0, 100, 100), 5)
    # Get the bounding box that contain all other smaller contours
    if contours:
        heading_line_x1, _, w1, _ = cv2.boundingRect(contours[0])
        heading_line_x2 = heading_line_x1 + w1
        for cnt in contours:
            _x1, _, _w1, _ = cv2.boundingRect(cnt)
            _x2 = _x1 + _w1
            heading_line_x1 = min(heading_line_x1, _x1)
            heading_line_x2 = max(heading_line_x2, _x2)

    line_len = heading_line_x2 - heading_line_x1
    if line_len < 1310 or line_len > 1328:
        print("There is some problem with the detection of heading line. Line length: ", line_len)

    if debug:
        cv2.line(page_image, (heading_line_x1, heading_line_y), (heading_line_x2, heading_line_y), (0, 255, 0), 3)
        cv2.imshow('heading line cut', head_line_cut)
        cv2.imshow('binary', binary)
        cv2.imshow('detected linesP', page_image)
        cv2.imshow('cropped', initial_head_cut)
        cv2.waitKey()

    # Return coordination's of heading line
    # Because the page has been deskewed before, just one y coordination will be returned
    return heading_line_x1, heading_line_y, heading_line_x2, heading_line_y


def find_entries_top_y(image_column, dividing_lines, search_band_width_right = 40, debug = False):
    """
    Entries are indented a little to the right side compared to the other parts of column. So by cutting a strip from the right side of the image, we can detect entries head words which show the starting point of entries.
    :param image_column: Input image column to detect entries in.
    :param search_band_width_right: Width of the search band from the right side of the image to detect entries.
    :param dividing_lines: Take the result of find_height_of_text_divider_lines() function as input.
    :param debug:
    :return: Height of detected entries in the input column of the page.
    """
    # Cut detection_strip from the right side of the image to detect entries
    detection_strip = image_column[:,-search_band_width_right:]
    # Clean detection strip
    detection_strip = denoise_open(detection_strip, 3, 1)
    detection_strip = contrast(detection_strip, 5, 210)
    kernel = np.ones((4, 4), np.uint8)
    detection_strip = cv2.dilate(cv2.bitwise_not(detection_strip), kernel, iterations=4)
    detection_strip = cv2.bitwise_not(detection_strip)

    contours, _ = cv2.findContours(cv2.bitwise_not(detection_strip), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Entry starts upper than the detected y of contour by amount of this value
    entries_y = []
    for cnt in contours:
        # _,y, _,h = cv2.boundingRect(cnt)
        # entries_y.append(y+h)
        M = cv2.moments(cnt)
        cy = int(M['m01'] / M['m00'])
        if floor_to_list(cy, dividing_lines) is not None:
            y = floor_to_list(cy, dividing_lines)
        else:
            warnings.warn("Entry detection failed. Check entry detection configs.")
            y = 0
        entries_y.append(y)

    if debug:
        cv2.drawContours(detection_strip, contours, -1, (0, 255, 0), 2)
        cv2.imshow('entry detection strip', detection_strip)
        cv2.waitKey()

    entries_y.sort()
    # Remove duplicates
    entries_y = list(dict.fromkeys(entries_y))

    return entries_y


def detect_entry_head_word(entry_first_line_image, min_area=160, bold_thresh=30, boldness_thresh = 1, width_thresh = 10, debug=False):
    """
    Detects the "Head Word" (main dictionary entry) by identifying bold text
    in the top-right region of an image entry (Optimized for RTL languages like Arabic/Persian).

    The function applies morphological erosion to filter out non-bold text,
    segments words, and checks if the right-most words intersect with the
    remaining bold 'blobs'.

    :param entry_first_line_image: Numpy array containing the image of the first line of the entry.
    :param min_area: Minimum pixel area to consider a blob as valid text (noise filter).
    :param bold_thresh: Threshold value for binarization after erosion.
                        Higher = strictly darker/bolder text required.
    :param debug: If True, visualizes the erosion steps and final detection.

    :return: A tuple (x1, y1, x2, y2) representing the bounding box of the head word.
             Returns None if detection fails.
    """

    # --- 2. Filter for Bold Text (Morphological Erosion) ---
    # Logic: Thin text disappears when eroded. Bold text shrinks but remains.

    # Invert image (Text=White, BG=Black) for correct morphological operation
    entry_first_line_image = remove_small_areas(entry_first_line_image, 20)
    cv2.imshow("entry_first_line_image", entry_first_line_image)

    img_inverted = cv2.bitwise_not(entry_first_line_image)


    skeleton = cv2.ximgproc.thinning(img_inverted, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # Erode: shrinks white regions. 4 iterations is aggressive, intended to wipe out normal fonts.
    kernel = np.ones((2, 2), np.uint8)
    eroded_img = cv2.erode(img_inverted, kernel, iterations=4)

    # Invert back to normal (Text=Black)
    eroded_img = cv2.bitwise_not(eroded_img)

    # Binarize to isolate the remaining "bold" blobs
    _, binary_bold_mask = cv2.threshold(eroded_img, bold_thresh, 255, cv2.THRESH_BINARY)

    # Helper function assumed to exist: removes tiny noise specks
    clean_bold_mask = remove_small_areas(binary_bold_mask, min_area)

    if debug:
        print("DEBUG: Visualizing bold text isolation steps.")
        cv2.imshow("1. Image of first line of entry", entry_first_line_image)
        cv2.imshow("2. Binary Bold Mask", binary_bold_mask)
        cv2.imshow("3. Eroded Image", eroded_img)
        cv2.imshow("4. Cleaned Mask", clean_bold_mask)
        cv2.imshow("5. Skeleton", skeleton)

    # --- 3. Word Segmentation ---
    # Get bounding boxes for all words in the header
    word_boxes = words_segmentation(entry_first_line_image, debug=False)

    if not word_boxes:
        warnings.warn("Head word detection failed: No words found in header.")
        return None

    # Ensure words are sorted Left-to-Right by x-coordinate
    word_boxes = sorted(word_boxes, key=lambda r: r[0])

    # --- 4. Identify Head Words (RTL Logic) ---
    # Assumption: In RTL languages (Arabic/Persian), the headword is on the far Right.
    detected_hw_boxes = []

    # The right-most word is assumed to be part of the head word
    right_most_word = word_boxes[-1]
    detected_hw_boxes.append(right_most_word)

    # Boldness = (Number of white pixels in the word) / (Number of white pixels in the skeleton of the word) [Both are inverted]
    # Calculate boldness of the right most word
    if debug:
        for i, (x1, y1, x2, y2) in enumerate(word_boxes):
            word_area = np.count_nonzero(img_inverted[y1:y2, x1:x2])
            word_path_length = skeleton_path_length(skeleton[y1:y2, x1:x2])
            boldness = word_area / word_path_length
            print(f"DEBUG: Word ({i}) boldness: {boldness}, area: {word_area}, path length: {word_path_length}")

    right_most_word_boldness = (np.count_nonzero(img_inverted[right_most_word[1]:right_most_word[3], right_most_word[0]:right_most_word[2]])/
                                np.count_nonzero(skeleton[right_most_word[1]:right_most_word[3], right_most_word[0]:right_most_word[2]]))

    # Check the second-to-last word (neighbor to the left).
    # If it overlaps with our "clean_bold_mask", it is also part of the head word.
    if len(word_boxes) > 1:
        second_to_last_word = word_boxes[-2]
        second_to_last_word_area = np.count_nonzero(entry_first_line_image[second_to_last_word[1]:second_to_last_word[3], second_to_last_word[0]:second_to_last_word[2]])
        #second_to_last_word_path_length = np.count_nonzero(skeleton[second_to_last_word[1]:second_to_last_word[3], second_to_last_word[0]:second_to_last_word[2]])
        second_to_last_word_path_length = skeleton_path_length(skeleton[second_to_last_word[1]:second_to_last_word[3], second_to_last_word[0]:second_to_last_word[2]])
        second_to_last_word_boldness = (second_to_last_word_area / second_to_last_word_path_length)
        second_to_last_word_width = second_to_last_word[2] - second_to_last_word[0]
        second_to_last_word_height = second_to_last_word[3] - second_to_last_word[1]

        right_most_word_area = np.count_nonzero(entry_first_line_image[right_most_word[1]:right_most_word[3], right_most_word[0]:right_most_word[2]])
        #right_most_word_path_length = np.count_nonzero(skeleton[right_most_word[1]:right_most_word[3], right_most_word[0]:right_most_word[2]])
        right_most_word_path_length = skeleton_path_length(skeleton[right_most_word[1]:right_most_word[3], right_most_word[0]:right_most_word[2]])
        right_most_word_boldness = (right_most_word_area / right_most_word_path_length)
        right_most_word_width = right_most_word[2] - right_most_word[0]
        right_most_word_height = right_most_word[3] - right_most_word[1]

        print("right most word , second to last word")
        print(f"area: {right_most_word_area}, {second_to_last_word_area}")
        print(f"path length: {right_most_word_path_length}, {second_to_last_word_path_length}")
        print(f"boldness: {right_most_word_boldness}, {second_to_last_word_boldness}")
        print(f"width: {right_most_word_width}, {second_to_last_word_width}")
        print(f"height: {right_most_word_height}, {second_to_last_word_height}")

        if abs(right_most_word_boldness - second_to_last_word_boldness) < boldness_thresh and \
           abs(right_most_word_width - second_to_last_word_width) < width_thresh:
            detected_hw_boxes.append(second_to_last_word)
            if debug: print("DEBUG: Multi-word headword detected.")

        # Check whether the second to last word boldness is close to the right most word boldness
        #if 1 - second_word_thresh < second_to_last_word_boldness/right_most_word_boldness < 1 + second_word_thresh:
        #    detected_hw_boxes.append(second_to_last_word)
        #    if debug: print("DEBUG: Multi-word headword detected.")

        #if is_rectangle_nonempty(second_to_last_word, clean_bold_mask):
        #    detected_hw_boxes.append(second_to_last_word)
        #    if debug: print("DEBUG: Multi-word headword detected.")

        # Check whether the area of the second to last word is close to the right most word
        #if 1 - second_word_thresh < second_to_last_word_area/right_most_word_area < 1 + second_word_thresh:
        #    detected_hw_boxes.append(second_to_last_word)
        #    if debug: print("DEBUG: Multi-word headword detected.")
    # --- 5. Merge Bounding Boxes ---
    # Calculate the bounding box that encompasses all detected head word parts
    x1_hw = min(rect[0] for rect in detected_hw_boxes)
    y1_hw = min(rect[1] for rect in detected_hw_boxes)
    x2_hw = max(rect[2] for rect in detected_hw_boxes)
    y2_hw = max(rect[3] for rect in detected_hw_boxes)

    head_word_rect = (x1_hw, y1_hw, x2_hw, y2_hw)

    if debug:
        img_debug = cv2.cvtColor(entry_first_line_image.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_debug, (x1_hw, y1_hw), (x2_hw, y2_hw), (0, 0, 255), 2)
        cv2.imshow("Final Head Word", img_debug)
        cv2.waitKey(0)  # Wait for key press to continue

    return head_word_rect


import math

def skeleton_path_length(skel):
    points = np.column_stack(np.where(skel > 0))
    skel_set = set(map(tuple, points))

    length = 0.0
    for y, x in skel_set:
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),
                       (-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = y+dy, x+dx
            if (ny, nx) in skel_set:
                length += math.sqrt(dx*dx + dy*dy)

    return length / 2  # avoid double counting



def words_segmentation(text_line_image, debug=False):
    """
    Segments a single line of text into individual words using Vertical Projection Profiles.

    The function converts the image to binary (white text on black background), computes
    the vertical sum of pixels, and identifies words by finding wide gaps (valleys)
    in the projection profile.

    :param text_line_image: A numpy array representing the input image (grayscale).
                            Must contain only a single line of text.
    :param debug: Boolean. If True, opens a window showing the detected bounding boxes.
    :return: A list of tuples, where each tuple represents a word's bounding box
             in the format (x1, y1, x2, y2).
    """
    # Threshold (make text white on black)
    _, binary = cv2.threshold(text_line_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary  # text=white

    # --- Vertical projection ---
    # Sum the white pixels in each column
    projection = np.sum(binary, axis=0)

    # --- Parameters ---
    # Noise barrier: consider anything below 5% of the max density as "empty space"
    proj_threshold = np.max(projection) * 0.05
    # Minimum width of a gap to be considered a word separator (prevents splitting letters)
    gap_min_length = 10

    # --- Detect candidate gap columns ---
    gaps = np.where(projection <= proj_threshold)[0]

    # --- Group continuous gaps into ranges ---
    gap_ranges = []
    if len(gaps) > 0:
        start = gaps[0]
        for i in range(1, len(gaps)):
            # If the current gap index is not consecutive, a gap range has ended
            if gaps[i] != gaps[i - 1] + 1:
                gap_ranges.append((start, gaps[i - 1]))
                start = gaps[i]
        gap_ranges.append((start, gaps[-1]))

    # --- Filter out short gaps ---
    # Only keep gaps that are wide enough to be spaces between words
    gap_ranges = [(s, e) for (s, e) in gap_ranges if (e - s + 1) >= gap_min_length]

    # --- Extract word boundaries ---
    words = []
    prev = 0
    words_rect = []

    # Everything between two gaps is a word
    for (s, e) in gap_ranges:
        if prev < s:
            x1, x2 = prev, s
            words.append((x1, x2))
        prev = e

    # Handle the last word after the final gap
    if prev < binary.shape[1]:
        words.append((prev, binary.shape[1]))

    # --- Draw rectangles around detected words ---
    text_line_image_copy = text_line_image.copy()
    for (x1, x2) in words:
        # Slice the binary image to find vertical limits (y1, y2)
        cols = binary[:, x1:x2]
        y_indices = np.where(cols > 0)[0]

        if len(y_indices) == 0:
            continue

        y1, y2 = y_indices.min(), y_indices.max()
        words_rect.append((x1, y1, x2, y2))

        if debug:
            cv2.rectangle(text_line_image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # --- Show result ---
    if debug:
        cv2.imshow("Words", text_line_image_copy)
        cv2.waitKey(0)

    # Convert np.int64 to standard int for JSON/Database compatibility
    words_rect = [(int(x1), int(y1), int(x2), int(y2)) for (x1, y1, x2, y2) in words_rect]

    return words_rect





if __name__ == '__main__':
    print("Head word detection test")
    entry_image = cv2.imread("/mnt/E0B80D7EB80D5506/sokhan/out/entries/0152_1_02.png")
    entry_image = cv2.cvtColor(entry_image, cv2.COLOR_BGR2GRAY)
    entry_first_line_image = entry_image[0:65, :]
    head_word_rect = detect_entry_head_word(entry_first_line_image, debug=True)
