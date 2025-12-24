from transforms import *
from detection import *
from text_line_segmentation import *
import warnings


def preprocess(image_page_input):
    """
    :param image_page_input: image of input page to be processed read by cv2.imread().
    :return: Preprocessed image

    This function does the following tasks on the input image file:
    1. Convert the input image to grayscale.
    2. Deskew the image to align the text lines horizontally.
    3. Detect the position of the heading line.
    4. Normalize the image size based on the width of the heading line.
    5. Crop the image to remove any unnecessary white space around the text.
    6. Redo the heading line detection to get a more accurate position.
    7. Perform an affine transformation to align the bottom text with the top text.
    8. Add a white band around the image to avoid any cropping issues in the future as safety margin.
    """

    image_page_input_gray = cv2.cvtColor(image_page_input, cv2.COLOR_BGR2GRAY)
    image_page_expand = expand_image_by_pixel(image_page_input_gray, 100)
    image_page_deskew, _ = deskew(image_page_expand, 600)
    coord_heading_line_step_1 = find_heading_line_position(image_page_deskew)

    if coord_heading_line_step_1 is None:
        warnings.warn("Preprocess failed. Check heading line detection configs.")
        return None

    heading_line_width = coord_heading_line_step_1[2] - coord_heading_line_step_1[0]
    heading_line_width_goal = 1320
    # Resize image to make a normalized image size based on heading line goal width
    image_page_normal = resize_image_by_ratio(image_page_deskew, heading_line_width_goal / heading_line_width)

    # The initial guess for text body is based on detected heading line position plus a constant value for height.
    text_body_height_initial_guess = 2155
    region_text_body_initial_guess = np.add(coord_heading_line_step_1, (0, 0, 0, 0 + text_body_height_initial_guess))
    # Remove area around text body based on detected heading line position to remove some noises that may be on the edges of pages with some safety margin
    image_page_clean, _ = extract_image_at_region(image_page_normal, region_text_body_initial_guess, (30, 10, 30, 10))

    # Add some white band around the image page to safely perform further processes.
    image_page_clean = expand_image_by_pixel(image_page_clean, 100, (1, 1, 1, 1))

    coord_heading_line_step_2 = find_heading_line_position(image_page_clean, heading_line_detection_height_up=50,
                                                             heading_line_detection_height_down=150)
    region_text_body_second_guess = np.add(coord_heading_line_step_2, (0, 0, 0, text_body_height_initial_guess))
    image_page_affine, _, _ = affine_transform(image_page_clean, region_text_body_second_guess)

    image_page_preprocessed_safety_margin = (10, 10, 20, 10)
    image_page_preprocessed, _ = extract_image_at_region(image_page_affine, region_text_body_second_guess,
                                                      image_page_preprocessed_safety_margin)

    image_page_preprocessed = expand_image_by_pixel(image_page_preprocessed, 100, (0, 0, 1, 0))
    image_page_preprocessed = image_page_preprocessed[:,
                              : heading_line_width_goal + image_page_preprocessed_safety_margin[0] +
                                image_page_preprocessed_safety_margin[2]]

    return image_page_preprocessed


def separate_columns(image_page_preprocessed):
    """
    :param image_page_preprocessed: This function gets the result of preprocess() function as input.
    :return: Separated right and left columns of the input image (Farsi starts from right).
    This function gets the preprocessed image and separate the right and left columns of the input page image.
    """
    # Extract left and right columns by using heading line coordination
    x1_hl, y1_hl, x2_hl, y2_hl = find_heading_line_position(image_page_preprocessed, heading_line_detection_height_up=0, heading_line_detection_height_down=50)

    image_column_left = image_page_preprocessed[:, :image_page_preprocessed.shape[1] // 2]
    image_column_right = image_page_preprocessed[:, image_page_preprocessed.shape[1] // 2:]

    # remove heading line from top of each column:
    image_column_left = image_column_left[y1_hl+15:,:]
    image_column_right = image_column_right[y1_hl+15:, :]

    # Remove empty spaces around each column and then expand to right and merge two column while the left sides of two columns aligned with each other.
    columns_safety_margin = (10, 12, 10, 10) # Change 2nd param if the height of starting dividing line is not correct.
    image_column_left = remove_empty_space_around_image(image_column_left, safety_margin = columns_safety_margin)
    image_column_right = remove_empty_space_around_image(image_column_right, safety_margin = columns_safety_margin)
    image_column_left = expand_image_by_pixel(image_column_left, image_page_preprocessed.shape[1] // 2 - image_column_left.shape[1], (0, 0, 1, 0))
    image_column_right = expand_image_by_pixel(image_column_right, image_page_preprocessed.shape[1] // 2 - image_column_right.shape[1], (0, 0, 1, 0))

    return image_column_right, image_column_left


def process_column(image_column,
                   entries_detection_search_width_band = 40,
                   dividing_lines_offset_y_start = 0,
                   dividing_lines_offset_y_end = 0):

    # Detected entries containers
    entries_rect = []
    entries_dividing_lines = []
    entries_type = []

    img_h, img_w = image_column.shape[:2]

    # --- 1. Detection Phase ---
    # Get absolute Y-coordinates of visual divider lines (e.g., horizontal rules)
    all_dividers = find_height_of_text_divider_lines(image_column, offset_y_start = dividing_lines_offset_y_start, offset_y_end = dividing_lines_offset_y_end)

    # Get absolute Y-coordinates where new entries logically begin
    entries_start_top_y = find_entries_top_y(image_column, all_dividers,
                                             search_band_width_right = entries_detection_search_width_band)

    # --- 2. Segmentation Phase ---

    # CASE A: No new entries detected on the whole page.
    # The entire column is treated as a continuation of the previous column/page.
    if len(entries_start_top_y) == 0:
        entries_rect.append((0, 0, img_w, img_h))
        entries_dividing_lines.append(all_dividers)
        entries_type.append("Continuation")

    else:
        # CASE B: Top Continuation.
        # If the first new entry has a meaningful gap from the top (half a line), the top chunk is a continuation.
        first_entry_y = entries_start_top_y[0]
        first_divider_y = all_dividers[0]

        if first_entry_y > first_divider_y:
            entries_rect.append((0, first_divider_y, img_w, first_entry_y))

            # Keep dividers that fall within this top chunk
            entries_dividing_lines.append([y for y in all_dividers if y <= first_entry_y])
            entries_type.append("Continuation")

        # CASE C: Standard Entries (Middle). In this case also the first entry (index = 0) is considered as a continuation.

        else:
            entries_rect.append(None)
            entries_dividing_lines.append(None)
            entries_type.append("Continuation")
        # Iterate through detected start points to extract entries between them.
        for i in range(len(entries_start_top_y) - 1):
            y_start = entries_start_top_y[i]
            y_end = entries_start_top_y[i + 1]

            entries_rect.append((0, y_start, img_w, y_end))

            # Extract dividers in this range and convert to relative coordinates
            # (subtract y_start so the line position is correct relative to the new crop)
            relative_dividers = [y - y_start for y in all_dividers if y_start <= y <= y_end]
            entries_dividing_lines.append(relative_dividers)

            entries_type.append("Entry")

        # CASE D: Final Entry.
        # Capture the segment from the last start point to the bottom of the image.
        last_y = entries_start_top_y[-1]

        if last_y != img_h:
            entries_rect.append((0, last_y, img_w, img_h))

            # Relative dividers for the bottom chunk
            entries_dividing_lines.append([y - last_y for y in all_dividers if y >= last_y])
            entries_dividing_lines[-1].append(img_h - last_y)
            entries_type.append("Entry")

    return all_dividers, entries_start_top_y, entries_rect, entries_dividing_lines, entries_type


def process_single_entry(image_column, entry_rect, entry_type, entry_relative_dividing_lines,
                         head_word_min_area = 160, head_word_bold_thresh = 30, boldness_thresh = 1, width_thresh = 10,
                         entry_vertical_offset = (0, 0), head_word_offset = (0, 0, 0, 0)):
    if entry_rect is None:
        # If entry_rect is None, it means it is a reserved space for a nonexisting continuation entry.
        return None, None, None, None

    # Extract entry image
    entry_top_offset, entry_bottom_offset = entry_vertical_offset

    entry_image, entry_rect_new = extract_image_at_region(image_column, entry_rect, (0, entry_top_offset, 0, entry_bottom_offset))

    entry_first_line_image = entry_image[0:entry_relative_dividing_lines[1]+entry_top_offset, :]



    if entry_type == "Entry":
        head_word_rect = detect_entry_head_word(entry_first_line_image, head_word_min_area, head_word_bold_thresh, boldness_thresh, width_thresh)
        head_word_image, head_word_rect_new = extract_image_at_region(entry_image, head_word_rect, head_word_offset)
    else:
        # If the entry is a continuation, there is no head word.
        head_word_image = None
        head_word_rect_new = None

    return entry_image, head_word_image, entry_rect_new, head_word_rect_new


def draw_guiding_lines(image_column, dividing_lines=None, entries_y=None, head_words_rect=None, entries_detection_search_width_band = 40):
    """
    This function is used to draw some lines on the image to help visualize the results of the process.
    """

    w, h = image_column.shape[1], image_column.shape[0]
    output_img = cv2.cvtColor(image_column, cv2.COLOR_GRAY2BGR)

    # Draw horizontal detected dividing lines between text lines
    if dividing_lines is not None:
        for y in dividing_lines:
            cv2.line(output_img, (0, y), (w, y), (255, 0, 255), 1)

    # Draw horizontal entries dividing lines where an entry detected
    if entries_y is not None:
        for y in entries_y:
            cv2.line(output_img, (0, y), (w, y), (255, 0, 0), 2)

    # Draw rectangle around detected head words
    if head_words_rect is not None:
        for i, head_word_rect in enumerate(head_words_rect):
            if head_word_rect is not None:
                x1 = head_word_rect[0]
                y1 = head_word_rect[1] + entries_y[i-1]
                x2 = head_word_rect[2]
                y2 = head_word_rect[3] + entries_y[i-1]
                cv2.rectangle(output_img, (x1,y1), (x2, y2), (0, 0, 255), 1)

    # Draw vertical entry detection line
    cv2.line(output_img, (w - entries_detection_search_width_band, 0),
                         (w - entries_detection_search_width_band, h), (0, 0, 255), 1)

    cv2.line(output_img, (10, 0), (10, h), (0, 0, 255), 1)

    return output_img


