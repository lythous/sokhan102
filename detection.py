from numpy.f2py.symbolic import normalize

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


def find_height_of_text_divider_lines_old_not_used(image, num_lines = 36, min_distance = 44, start_band = 30, end_band = 30, offset_y_start = 0, offset_y_end = 0):
    """
    Find the optimal dividing lines between text lines on a single-column page image.
    :param image: Grayscale image as numpy array
    :param  num_lines: Number of text lines in the image
    :param  min_distance: Minimum vertical distance between lines
    :param  start_band: (start_y_min, start_y_max) range for first line's top band
    :param  end_band: (end_y_min, end_y_max) range for last line's bottom band
    :param  offset_y_start: Offset for first line's top band. Increase to move the first line down.
    :param  offset_y_end: Offset for last line's bottom band. Increase to move the last line down.
    :return: List of y coordinates (integers) representing the dividing lines between text lines
    """
    height, width = image.shape

    # Binarize the image: text is black (0), background is white (255)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define kernel for dilation (adjust size for boldness)
    kernel = np.ones((2, 2), np.uint8)

    # Dilate to make text thicker
    dilated = cv2.dilate(binary, kernel, iterations=2)


    min_total_black = float('inf')
    best_dividers = []

    for start_y in range(0, start_band):
        for end_y in range(end_band, height+1):
            available_height = end_y - start_y
            if available_height < min_distance * (num_lines - 1):
                continue

            line_distance = available_height / (num_lines - 1)
            if line_distance < min_distance:
                continue

            dividers = [int(round(start_y + i * line_distance)) for i in range(num_lines+1)]

            # Compute a sum of black pixels on each dividing line (horizontal scantiness)
            total_black = sum(np.count_nonzero(dilated[y, :] > 0) for y in dividers if 0 <= y < height)

            if total_black < min_total_black:
                min_total_black = total_black
                best_dividers = dividers

    # Offset the dividers based on the provided values
    total_offset = offset_y_end - offset_y_start
    lines_offset = [offset_y_start + int(i*total_offset/num_lines) for i in range(num_lines)]
    best_dividers_offset = [y + offset for y, offset in zip(best_dividers, lines_offset)]
    best_dividers_offset = [min(max(y, 0), height) for y in best_dividers_offset]

    return best_dividers_offset


def find_height_of_text_divider_lines(gray_image, no_of_text_lines = 36, debug=False):
    # Binarize the image: text is black (0), background is white (255)
    # denoised = denoise_erode(gray_image, 1, 1)

    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    height, width = binary.shape

    # Count non-zero pixels per row
    non_zero_per_row = np.count_nonzero(binary, axis=1)

    white_gap_threshold = 0.01

    # Find rows with less than white_gap_threshold width activity
    white_rows = np.where(non_zero_per_row < (width * white_gap_threshold))[0]

    white_rows = white_rows.tolist()

    white_rows_evenly_spaced = find_best_dividers(white_rows, no_of_text_lines + 1)
    print("white_rows_evenly_spaced: ", white_rows_evenly_spaced)



    if debug:

        debug_image_least_active_rows = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for y in white_rows:
            cv2.line(debug_image_least_active_rows, (0, y), (width, y), (255, 0, 255), 1)
        cv2.imshow("All least active rows", debug_image_least_active_rows)

        debug_image_least_active_rows_evenly_spaced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for y in white_rows_evenly_spaced:
            cv2.line(debug_image_least_active_rows_evenly_spaced, (0, y), (width, y), (255, 0, 255), 1)
        cv2.imshow("Least active rows evenly spaced", debug_image_least_active_rows_evenly_spaced)
        cv2.imwrite("debug_image_least_active_rows_evenly_spaced.png", debug_image_least_active_rows_evenly_spaced)
        cv2.waitKey()




def find_best_dividers(white_rows, num_dividers = 37):
    """
    Identifies exactly 37 dividing lines from a list of white row indices
    by clustering the inputs and fitting an optimal grid.
    """
    if not white_rows:
        return []

    # --- Step 1: Clustering ---
    # The input is raw row indices. Consecutive numbers (e.g., 52, 53, 54)
    # represent the same physical gap. We group them into 'clusters'.
    clusters = []
    current_cluster = [white_rows[0]]

    for i in range(1, len(white_rows)):
        # If the gap is small (<= 5 pixels), it belongs to the same cluster
        if white_rows[i] - white_rows[i - 1] <= 5:
            current_cluster.append(white_rows[i])
        else:
            # Calculate the median of the finished cluster
            clusters.append(int(np.median(current_cluster)))
            current_cluster = [white_rows[i]]
    clusters.append(int(np.median(current_cluster)))

    # We now have N candidate positions (likely 40-50).
    # We need to select exactly 37.

    # --- Step 2: Global Grid Search ---
    # We test every possible 'Start' and 'End' line pair.
    # For each pair, we construct a perfect grid and see how well
    # the actual clusters match this grid.

    best_selection = []
    min_total_error = float('inf')


    n = len(clusters)

    # Iterate over all possible first lines
    for i in range(n - num_dividers + 1):
        # Iterate over all possible last lines
        for j in range(i + num_dividers - 1, n):

            start_y = clusters[i]
            end_y = clusters[j]

            # Calculate the ideal step size for this specific start/end pair
            total_height = end_y - start_y
            step = total_height / (num_dividers - 1)

            # Optimization: Sanity check the step size
            # If text lines are usually 40-80px, ignore absurd grids
            if step < 20 or step > 150:
                continue

            # Generate the 37 ideal points
            ideal_grid = [start_y + k * step for k in range(num_dividers)]

            # Match each ideal point to the closest available cluster
            current_selection = []
            current_error = 0
            possible = True

            cluster_idx = i  # Start searching from the current start cluster

            for ideal_point in ideal_grid:
                # Find the cluster closest to this ideal_point
                # We scan forward from the last used cluster to save time
                best_dist = float('inf')
                best_match = -1

                # Look ahead in a reasonable window to find the match
                while cluster_idx < n:
                    dist = abs(clusters[cluster_idx] - ideal_point)

                    if dist < best_dist:
                        best_dist = dist
                        best_match = clusters[cluster_idx]
                        # If we are moving away from the ideal point, stop scanning this cluster
                        # (since the list is sorted)
                    elif dist > best_dist:
                        break

                    cluster_idx += 1

                # Backtrack index slightly for the next iteration
                # (in case clusters are very close)
                cluster_idx = max(0, cluster_idx - 2)

                # Accumulate error (Squared Distance)
                current_error += best_dist ** 2
                current_selection.append(best_match)

                # Heuristic: If a single line is too far off (> step/2),
                # this grid is invalid (we probably skipped a line)
                if best_dist > step * 0.5:
                    possible = False
                    break

            if possible and current_error < min_total_error:
                min_total_error = current_error
                best_selection = current_selection

    return best_selection


def cluster_consecutive_centers(nums):
    """
    Groups consecutive integers and returns the integer center of each group.
    Example:

    nums = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127,
    132,
    170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186
    ]

    result = [5, 60, 117, 132, 178]
    """
    if not nums:
        return []

    nums = sorted(nums)
    clusters = []
    start = prev = nums[0]

    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            clusters.append((start, prev))
            start = prev = n

    clusters.append((start, prev))

    # Compute integer centers
    centers = [(a + b) // 2 for a, b in clusters]
    return centers


def get_most_evenly_spaced_lines(centers):
    """
    Selects a subsequence of lines that are most evenly spaced using
    Dynamic Programming (Viterbi-style shortest path).
    """
    if not centers:
        return []

    # 1. Estimate the "Ideal Step" (Global Average)
    # We look at the median of differences > 10 to ignore intra-cluster noise
    diffs = np.diff(centers)
    valid_diffs = diffs[diffs > 10]
    if len(valid_diffs) == 0: return centers

    approx_median = np.median(valid_diffs)

    # Calculate how many lines effectively fit in the total span
    total_span = centers[-1] - centers[0]
    estimated_num_gaps = round(total_span / approx_median)

    # The precise float step that would result in perfect spacing
    ideal_step = total_span / estimated_num_gaps

    n = len(centers)

    # 2. Dynamic Programming Setup
    # min_cost[i]: The lowest accumulated error to reach center[i]
    min_cost = [float('inf')] * n
    parent = [-1] * n

    # Start at the first item
    min_cost[0] = 0

    # We restrict search to a reasonable window to speed up
    # (0.5x to 1.5x the ideal step)
    min_jump = ideal_step * 0.5
    max_jump = ideal_step * 1.5

    # 3. Build the Graph
    for i in range(n):
        if min_cost[i] == float('inf'):
            continue

        for j in range(i + 1, n):
            jump = centers[j] - centers[i]

            # Optimization: distinct lines shouldn't be too close
            if jump < min_jump:
                continue
                # Optimization: if jump is too big, we skipped a line, stop checking further
            if jump > max_jump:
                break

                # Cost function: Squared Error from the ideal step
            # Using squared error penalizes large deviations heavily
            step_cost = (jump - ideal_step) ** 2

            total_cost = min_cost[i] + step_cost

            if total_cost < min_cost[j]:
                min_cost[j] = total_cost
                parent[j] = i

    # 4. Backtrack to find the optimal path
    path = []
    curr = n - 1  # Start from the last element

    # If the last element wasn't reachable (rare edge case), scan backwards
    while curr != -1 and min_cost[curr] == float('inf'):
        curr -= 1

    while curr != -1:
        path.append(centers[curr])
        curr = parent[curr]

    return path[::-1]  # Reverse to get start-to-end


def place_dividing_lines(text_lines, num_lines, min_distance, image_height):
    """
    Places N dividing lines between text lines with minimum overlap and enforced spacing.

    Parameters:
        text_lines:     list of (top, bottom) y-coordinates of text blocks
        num_lines:      number of dividing lines required
        min_distance:   minimum distance between dividing lines
        image_height:   image bottom limit (fallback for final line)

    Returns:
        list of Y coordinates of dividing lines
    """

    # 1. Compute all candidate gaps between lines (midpoints)
    gaps = []
    for i in range(len(text_lines) - 1):
        top = text_lines[i][1]
        bottom = text_lines[i + 1][0]
        gap_mid = (top + bottom) // 2
        overlap_score = max(0, text_lines[i][1] - text_lines[i + 1][0])  # zero if valid gap
        gaps.append((gap_mid, overlap_score))

    # If no gaps found, fallback to evenly spaced lines
    if not gaps:
        step = image_height // (num_lines + 1)
        return [(i + 1) * step for i in range(num_lines)]

    # 2. Sort gaps by overlap (lower is better), then by how centered the gap is
    gaps.sort(key=lambda g: (g[1], g[0]))

    selected = []

    for candidate, _ in gaps:
        if len(selected) == num_lines:
            break

        # 3. Check spacing constraint
        if all(abs(candidate - s) >= min_distance for s in selected):
            selected.append(candidate)

    # If we still have too few dividing lines, fill missing ones
    while len(selected) < num_lines:
        # Add next best fallback: place below last or evenly if empty
        if selected:
            new_line = selected[-1] + min_distance
            new_line = min(new_line, image_height - 1)
        else:
            step = image_height // (num_lines + 1)
            new_line = step

        if all(abs(new_line - s) >= min_distance for s in selected):
            selected.append(new_line)
        else:
            # ultimate fallback: reduce spacing
            new_line = selected[-1] + (min_distance // 2)
            selected.append(min(new_line, image_height - 1))

    # 4. Sort final output and add last dividing line at bottom if needed
    selected = sorted(selected)

    # Add final dividing line if last gap is missing
    if selected[-1] < image_height - min_distance:
        selected.append(image_height - 1)

    return selected


def find_height_of_text_divider_lines_old(
        image,
        num_lines=37,
        min_distance=44,
        start_band=30,
        end_band=30,
        offset_y_start=0,
        offset_y_end=0
    ):
    """
    Fast, clear algorithm to determine horizontal dividing lines between text lines
    in a single-column scanned page.

    Improvements:
        - Much faster (precomputes per-row black pixels)
        - Clear comments and simplified logic
        - Guarantees the last divider below final text
        - If not found, appends the image height

    Returns:
        List[int] of divider y-positions (length = num_lines + 1)
    """

    height, width = image.shape

    # --- STEP 1: Binarize + Dilate (to make text thicker)
    _, binary = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Slight dilation for robust gap detection
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # --- STEP 2: Compute black pixel count per row (O(height×width) done once)
    # dilated: text=white(255) → 1, background=black(0) → 0
    row_black_count = np.count_nonzero(dilated, axis=1)

    # --- Optimization search
    best_total_black = float("inf")
    best_dividers = None

    # Candidate ranges for start and end
    start_candidates = range(0, start_band + 1)
    end_candidates   = range(height - end_band, height + 1)

    # --- STEP 3: Try all valid start/end combinations
    for start_y in start_candidates:
        for end_y in end_candidates:

            available_height = end_y - start_y
            min_required_height = min_distance * (num_lines - 1)

            # Skip impossible configurations
            if available_height < min_required_height:
                continue

            # Ideal uniform spacing
            line_dist = available_height / (num_lines - 1)
            if line_dist < min_distance:
                continue

            # Generate divider grid
            divs = [int(round(start_y + i * line_dist)) for i in range(num_lines)]

            # Skip out-of-bounds dividers
            if divs[0] < 0 or divs[-1] >= height:
                continue

            # Cost = sum of black pixels on each divider
            total_black = sum(row_black_count[y] for y in divs)

            if total_black < best_total_black:
                best_total_black = total_black
                best_dividers = divs

    # --- STEP 4: If no configuration was found, fall back to uniform spacing
    if best_dividers is None:
        line_dist = height / num_lines
        best_dividers = [int(i * line_dist) for i in range(num_lines)]

    # --- STEP 5: Apply linear offset across all divider lines
    total_offset = offset_y_end - offset_y_start
    per_line_offset = [
        offset_y_start + int(i * total_offset / (num_lines - 1))
        for i in range(num_lines)
    ]

    best_dividers = [
        min(max(y + dy, 0), height)
        for y, dy in zip(best_dividers, per_line_offset)
    ]

    # --- STEP 6: Add final bottom divider:
    # 1) Try to place it below last text row
    # 2) If no text found, default to image height
    last_text_row_candidates = np.where(row_black_count > 0)[0]
    if len(last_text_row_candidates) > 0:
        last_text_y = int(last_text_row_candidates[-1])
        last_divider = min(last_text_y + min_distance, height)
    else:
        last_divider = height

    # Append final bottom divider
    best_dividers.append(last_divider)

    return best_dividers


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


def detect_head_word(image_column, entries_y, dividing_lines, min_area, bold_thresh, debug = False):
    head_words_rect = []
    for i, y in enumerate(entries_y):

        y_next_line = dividing_lines[dividing_lines.index(y) + 1]
        entry_image_top_line = image_column[y:y_next_line, :]

        # --- Filter bold text (head words) ---
        # By using erode and filter I'm trying to detect just bold text which are hopefully head words.
        kernel = np.ones((2, 2), np.uint8) # Filter parameter 1. To be changed for best results.
        erode = cv2.erode(cv2.bitwise_not(entry_image_top_line), kernel, iterations=4)  # Filter parameter 2. To be changed for best results.
        erode = cv2.bitwise_not(erode)
        _, binary = cv2.threshold(erode, bold_thresh, 255, cv2.THRESH_BINARY) # Filter parameter 3. More = Less sensitive
        filtered_head_word = remove_small_areas(binary, min_area) # Filter parameter 4. To be changed for best results.

        if debug:
            print(f"Detecting head word region for entry {i}")
            cv2.imshow("Head word binary", binary)
            cv2.imshow("Head word erode", erode)
            cv2.imshow("Head word filtered (small black areas removed)", filtered_head_word)


        # Find those words in the original page image that overlap with filtered head words regions.
        #if find_containing_region(filtered_head_word) is not None:
        #    x1_hw_fil, _, x2_hw_fil, _ = find_containing_region(filtered_head_word)
        #else:
        #    x1_hw_fil = entry_image_top_line.shape[1]
        #    x2_hw_fil = x1_hw_fil
        #    print("Head word filtering parameters are too harsh.")

        words_rect = words_segmentation(entry_image_top_line, debug)
        # The rightest word is always a part of head word
        head_word_rects = [words_rect[-1]]
        # The main question is whether the one word before last right word is also a part of head word or not?
        # In other words, whether the head word is two word or single work?
        if len(words_rect) > 1:
            if is_rectangle_nonempty(words_rect[-2], filtered_head_word):
                head_word_rects.append(words_rect[-2])
                print(f"Entry {i} has two head words.")
        else:
            warnings.warn(f"Head word detection for entry {i} has failed.")

        # Find the rectangle that contain all other rectangles.
        x1_hw, y1_hw, x2_hw, y2_hw = head_word_rects[0]
        head_word_rect = (x1_hw, y1_hw, x2_hw, y2_hw)
        for rect in head_word_rects:
            x1, y1, x2, y2 = rect
            x1_hw = min(x1_hw, x1)
            y1_hw = min(y1_hw, y1)
            x2_hw = max(x2_hw, x2)
            y2_hw = max(y2_hw, y2)

        # Offset head word rectangle height to the entry height
        head_word_rect = (x1_hw, y1_hw+y, x2_hw, y2_hw+y)

        if debug:
            # Draw rectangle around head words
            img_debug = entry_image_top_line.copy()
            img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img_debug, (x1_hw, y1_hw), (x2_hw, y2_hw), (0, 0, 255), 1)
            cv2.imshow("Head word detection box", img_debug)
            cv2.waitKey()


        head_words_rect.append(head_word_rect)
    return head_words_rect


def detect_entry_head_word(entry_first_line_image, min_area=160, bold_thresh=30, debug=False):
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
    img_inverted = cv2.bitwise_not(entry_first_line_image)

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
        cv2.imshow("1. Binary Bold Mask", binary_bold_mask)
        cv2.imshow("2. Eroded Image", eroded_img)
        cv2.imshow("3. Cleaned Mask", clean_bold_mask)

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

    # Check the second-to-last word (neighbor to the left).
    # If it overlaps with our "clean_bold_mask", it is also part of the head word.
    if len(word_boxes) > 1:
        second_to_last_word = word_boxes[-2]
        if is_rectangle_nonempty(second_to_last_word, clean_bold_mask):
            detected_hw_boxes.append(second_to_last_word)
            if debug: print("DEBUG: Multi-word headword detected.")

    # --- 5. Merge Bounding Boxes ---
    # Calculate the bounding box that encompasses all detected head word parts
    x1_hw = min(rect[0] for rect in detected_hw_boxes)
    y1_hw = min(rect[1] for rect in detected_hw_boxes)
    x2_hw = max(rect[2] for rect in detected_hw_boxes)
    y2_hw = max(rect[3] for rect in detected_hw_boxes)

    head_word_rect = (x1_hw, y1_hw, x2_hw, y2_hw)

    if debug:
        img_debug = cv2.cvtColor(entry_header_crop.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_debug, (x1_hw, y1_hw), (x2_hw, y2_hw), (0, 0, 255), 2)
        cv2.imshow("Final Head Word", img_debug)
        cv2.waitKey(0)  # Wait for key press to continue

    return head_word_rect


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