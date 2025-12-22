import numpy as np
import cv2


def emphasize_high_sigmoid(values, steepness=10):
    values = np.asarray(values)
    y = 1 / (1 + np.exp(-steepness * (values - 0.5)))
    return (y - y.min()) / (y.max() - y.min())


def horizontal_projection_profile_normalized(binary_image):
    height, width = binary_image.shape
    # Count non-zero pixels per row
    projection = np.count_nonzero(binary_image, axis=1)
    projection_normalized = projection / width
    return projection_normalized


def find_height_of_text_divider_lines_new(gray_image, no_of_text_lines = 36, y0_search_factor = 0.5, scale_search_factor = 0.2, scale_search_step = 0.002, offset_y_start = 0, offset_y_end = 0, debug=False):
    height, width = gray_image.shape

    # Binarize the image: text is black (0), background is white (255)
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate:
    #kernel = np.ones((2, 2), np.uint8)
    #binary = cv2.dilate(binary, kernel, iterations=5)

    density = horizontal_projection_profile_normalized(binary)
    #density = emphasize_high_sigmoid(density)

    density_gray = (density * 255).astype(np.uint8)

    print(density_gray)
    _, _, _, lines = find_best_line_grid(density, no_of_text_lines, np.arange(0, int(y0_search_factor * height // no_of_text_lines)), np.arange(1 - scale_search_factor, 1 + scale_search_factor, scale_search_step))
    lines = readjust_dividing_lines(lines, density)

    # Offset the dividers based on the provided values
    total_offset = offset_y_end - offset_y_start
    lines_offset = [offset_y_start + int(i*total_offset/no_of_text_lines) for i in range(no_of_text_lines)]
    lines = [y + offset for y, offset in zip(lines, lines_offset)]
    lines = [min(max(y, 0), height) for y in lines]

    mid_lines = [(lines[i] + lines[i + 1]) // 2 for i in range(len(lines) - 1)]
    print(lines)


    if debug:
        # Colorize original image
        original_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        for y in lines:
            cv2.line(original_image, (0, y), (width, y), (255, 0, 255), 1)

        #for y in mid_lines:
        #    cv2.line(original_image, (0, y), (width, y), (0, 255, 0), 1)

        cv2.imshow("Original", original_image)
        cv2.imwrite("original_image.png", original_image)
        # Visualize the density profile
        # Expand to 0.2 of image width
        density_image = np.repeat(density_gray[:, None], width//5, axis=1)


        cv2.imshow("Density", density_image)
        cv2.waitKey()


def find_best_line_grid(density_profile,
                        num_text_lines,
                        y0_range,
                        scale_range):
    """
    Finds y0 and scale that minimize summed density at sampled line positions.

    Parameters:
        density_profile (array-like): row density values
        num_text_lines (int): number of horizontal lines
        y0_range (np.ndarray): starting y candidates
        scale_range (np.ndarray): spacing scale candidates

    Returns:
        best_y0 (float)
        best_scale (float)
        min_total_density (float)
        best_line_positions (np.ndarray)
    """

    density = np.asarray(density_profile)
    height = len(density)

    base_spacing = height / (num_text_lines + 1)

    min_total = np.inf
    best_y0 = None
    best_scale = None
    best_lines = None

    for y0 in y0_range:
        for scale in scale_range:
            # Compute line positions
            line_positions = y0 + base_spacing * scale * np.arange(num_text_lines)

            # Convert to integer indices
            line_indices = np.round(line_positions).astype(int)

            # Keep valid indices only
            if np.any(line_indices < 0) or np.any(line_indices >= height):
                continue

            total_density = density[line_indices].sum()

            if total_density < min_total:
                min_total = total_density
                best_y0 = y0
                best_scale = scale
                best_lines = line_indices

    return best_y0, best_scale, min_total, best_lines



def readjust_dividing_lines(dividing_lines, density, search_range = 10):
    """
    This function readjusts the detected dividing lines based on the density profile by searchin in a search range
    around each detected dividing lines separately so that they are placed at the minimum of the density profile.
    """

    for i, y in enumerate(dividing_lines):
        for j in range(search_range):
            if y + j < len(density) and density[y + j] < density[y]:
                dividing_lines[i] = y + j
            if y - j >= 0 and density[y - j] < density[y]:
                dividing_lines[i] = y - j
    return dividing_lines


if __name__ == '__main__':
    print("Sokhan Processor Test")
    image = cv2.imread("/mnt/E0B80D7EB80D5506/sokhan/out/columns/0150_0.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    find_height_of_text_divider_lines_new(image, debug=True)
