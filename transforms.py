import warnings
from auxiliaries import *

def expand_to_right(img, expansion_length):
    h, w = img.shape[:2]
    white_frame = np.zeros([h, expansion_length,3],dtype=np.uint8)
    white_frame.fill(255)
    expanded_image = np.concatenate((img, white_frame), axis=1)
    return expanded_image


def expand_image_by_ratio(image, expansion_ratio, expand_sides = (0, 1, 1 , 0),expansion_color = (255, 255, 255)):
    height, width = image.shape[:2]
    top_border = int(height * expansion_ratio) * expand_sides[0]
    bottom_border = int(height * expansion_ratio) * expand_sides[1]
    left_border = int(width * expansion_ratio) * expand_sides[2]
    right_border = int(width * expansion_ratio) * expand_sides[3]

    # Add a white border around the image
    bordered_image = cv2.copyMakeBorder(
        image, top_border, bottom_border, left_border, right_border,
        cv2.BORDER_CONSTANT, value=expansion_color
    )
    return bordered_image

def expand_image_by_pixel(image, expansion_width_pixel, expand_sides = (1, 0, 0 , 1),expansion_color = (255, 255, 255)):
    height, width = image.shape[:2]
    left_border = expansion_width_pixel * expand_sides[0]
    top_border = expansion_width_pixel * expand_sides[1]
    right_border = expansion_width_pixel * expand_sides[2]
    bottom_border = expansion_width_pixel * expand_sides[3]

    # Add a white border around the image
    bordered_image = cv2.copyMakeBorder(
        image, top_border, bottom_border, left_border, right_border,
        cv2.BORDER_CONSTANT, value=expansion_color
    )
    return bordered_image



def get_skew_angle(img, threshold, res, search_band):
    # Use edge detection to find the edges in the image
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # Apply Hough Line Transform to detect lines in the image
    lines = cv2.HoughLines(edges,1,np.pi*res/180,threshold, min_theta = np.pi/2-search_band*np.pi/180, max_theta = np.pi/2+search_band*np.pi/180)
    # Calculate the angles of all the detected lines
    angles = []
    if lines is None:
        warnings.warn("Deskew unsuccessful! Change line detection parameters.")
        return 0
    for line in lines:

        _, theta = line[0]

        angle = np.degrees(theta)
        angles.append(angle)
    # Compute the median angle
    average_angle = np.average(angles)

    return 90.0 - average_angle

def rotate_image(img, angle):
    # Get the image dimensions
    (h, w) = img.shape[:2]

    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def deskew(image_page, threshold, debug=0):
    # Calculate the skew angle from a cut of the up of the source page that contain heading line.
    # This two constant must be so valued that in all pages cut band contain heading line.
    heading_line_detection_upper_bound = 50
    heading_line_detection_lower_bound = 250

    heading_line_detection_strip = image_page[heading_line_detection_upper_bound:heading_line_detection_lower_bound]
    blur = cv2.blur(heading_line_detection_strip, (5,5))
    _, binary = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY)

    angle = get_skew_angle(heading_line_detection_strip, threshold, 0.1, 10)
    # Rotate the image to align the text horizontally
    rotated = rotate_image(image_page, -angle)
    if debug:
        print('deskew angle', angle)
        cv2.imshow('deskewed image', rotated)
        cv2.waitKey()
    return rotated, angle


def affine_transform(page_image, text_body_reference_points, corner_crop_ratio = 0.3, bottom_offset=0):
    # This function consider upper text body points as reference and move bottom text to align vertically with it.
    # Input page image must deskewed before calling this function.
    # Deskewing is necessary to ensure upper text body points are horizontal and fixed.
    x1, y1, x2, y2 = text_body_reference_points

    # Crop a piece of input page image to recognize bottom left corner of the text
    h, w = page_image.shape[:2]
    corner_crop_h = int(h * corner_crop_ratio)
    corner_crop_w = int(w * corner_crop_ratio)


    bottom_left_crop = page_image[h-corner_crop_h:h, 0:corner_crop_w]
    bottom_left_crop = contrast(bottom_left_crop, 5, 210)
    contours, _ = cv2.findContours(cv2.bitwise_not(bottom_left_crop), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours == ():
        print("Affine transform failed. Check its config.")
        return None



    x_bottom_left_unaligned,_,_,_ = find_rectangle_contain_all_contours(contours)
    #print("x bottom left unaligned: ", x_bottom_left_unaligned)
    #x_bottom_left_unaligned += x1
    #print("x bottom left unaligned: ", x_bottom_left_unaligned)

    src_pts = np.float32([[x1, y1],
                          [x2, y1],
                          [x_bottom_left_unaligned, y2]])

    dst_pts = np.float32([[x1, y1],
                          [x2, y1],
                          [x1+bottom_offset, y2]])

    rotation_matrix = cv2.getAffineTransform(src_pts, dst_pts)
    page_image_affine_transform = cv2.warpAffine(page_image, rotation_matrix, (w, h))

    # Draw lines on the input page for debug
    before_affine_debug = page_image.copy()
    cv2.line(before_affine_debug, (x1, y1), (x2, y1), (0, 255, 0), 3)
    cv2.line(before_affine_debug, (x1, y1), (x_bottom_left_unaligned, y2), (0, 0, 255), 3)
    cv2.line(before_affine_debug, (x2, y1), (x_bottom_left_unaligned, y2), (0, 0, 255), 3)

    after_affine_debug = page_image_affine_transform.copy()
    cv2.line(after_affine_debug, (x1, y1), (x2, y1), (0, 255, 0), 3)
    cv2.line(after_affine_debug, (x1, y1), (x1 + bottom_offset, y2), (0, 0, 255), 3)
    cv2.line(after_affine_debug, (x2, y1), (x1 + bottom_offset, y2), (0, 0, 255), 3)

    return page_image_affine_transform, before_affine_debug, after_affine_debug



""" May be deprecated """
def affine_transform_old(page_image, region_of_interest, bottom_offset=0):
    x1, y1, x2, y2 = region_of_interest

    # Crop a piece of img_proc to recognize bottom left corner of the text
    h, w = page_image.shape[:2]
    strip_w = 80
    strip_h = 400
    print('page image shape: ',page_image.shape)
    if x1 < strip_w//2:
        strip_w = 2*x1
    bottom_left_crop = page_image[y2-strip_h:y2, x1-strip_w//2:x1+strip_w//2]
    bottom_left_crop = contrast(bottom_left_crop, 5, 210)


    contours, _ = cv2.findContours(cv2.bitwise_not(bottom_left_crop), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours == ():
        print("Affine transform failed. Check its config.")
        return None
    # Show contours for debug
    #cv2.drawContours(img, contours, -1, (0, 100, 100), 30)
    #cv2.imshow('contours', img)


    x_bottom_left,_,_,_ = find_rectangle_contain_all_contours(contours)
    print('x_bottom_left:', x_bottom_left)
    x_bottom_left += x1

    src_pts = np.float32([[x1, y1],
                          [x2, y1],
                          [x_bottom_left-strip_w//2, y2]])

    dst_pts = np.float32([[x1, y1],
                          [x2, y1],
                          [x1+bottom_offset, y2]])

    rotation_matrix = cv2.getAffineTransform(src_pts, dst_pts)
    page_image_affine_transform = cv2.warpAffine(page_image, rotation_matrix, (w, h))

    # Draw lines on the input page for debug
    before_affine_debug = page_image.copy()
    cv2.line(before_affine_debug, (x1, y1), (x2, y1), (0, 255, 0), 3)
    cv2.line(before_affine_debug, (x1, y1), (x_bottom_left - strip_w // 2, y2), (0, 0, 255), 3)
    cv2.line(before_affine_debug, (x2, y1), (x_bottom_left - strip_w // 2, y2), (0, 0, 255), 3)

    after_affine_debug = page_image_affine_transform.copy()
    cv2.line(after_affine_debug, (x1, y1), (x2, y1), (0, 255, 0), 3)
    cv2.line(after_affine_debug, (x1, y1), (x1 + bottom_offset, y2), (0, 0, 255), 3)
    cv2.line(after_affine_debug, (x2, y1), (x1 + bottom_offset, y2), (0, 0, 255), 3)

    return page_image_affine_transform, before_affine_debug, after_affine_debug