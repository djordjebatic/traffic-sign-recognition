import numpy as np
import cv2


def adjust_gamma(image, gamma=1):
    """automation gamma value adjustment.
    Args:
        image: source image.
        gamma: gamma value.
    Return:
        gamma corrected image.
    """
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def auto_canny(image, sigma=0.33):
    """apply canny edge detection on image.
    Args:
        image: source image.
        sigma: sigma value.
    Return:
        edged: a binary image.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def filter_sign_color(color, img_bgr):
    """preprocess the image for contour detection.
    Args:
        color: desired filter [yellow, blue, red].
        img_bgr: bgr image on which filter is applied.
    Return:
        img_bin: a binary image.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    low = np.array([0, 42, 0])
    high = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, low, high)
    hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    adjusted_hsv = hsv

    if color == 'yellow':
        # yellow hsv mask
        blue_lower = np.array([17, 100, 60], np.uint8)
        blue_upper = np.array([30, 255, 255], np.uint8)
        img_ybin = cv2.inRange(adjusted_hsv, blue_lower, blue_upper)

        return img_ybin

    elif color == 'blue':
        # blue hsv mask
        blue_lower = np.array([100, 130, 0], np.uint8)
        blue_upper = np.array([140, 255, 255], np.uint8)
        img_bbin = cv2.inRange(adjusted_hsv, blue_lower, blue_upper)

        return img_bbin

    elif color == 'red':
        # lower red hsv mask
        lower_red = np.array([0, 80, 0])
        upper_red = np.array([12, 255, 255])
        mask0 = cv2.inRange(adjusted_hsv, lower_red, upper_red)
        # upper red hsv mask
        lower_red = np.array([140, 70, 0])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(adjusted_hsv, lower_red, upper_red)

        img_rbin = np.maximum(mask0, mask1)

        return img_rbin


def preprocess_img(color, img_bgr, erode_dilate=True):
    """preprocess the image for contour detection.
    Args:
        img_bgr: source image.
        erode_dilate: erode and dilate or not.
    Return:
        img_bin: a binary image (blue and red).
    """
    rows, cols, _ = img_bgr.shape
    img_bin = filter_sign_color(color, img_bgr)

    if erode_dilate is True:
        kernel_erosion = np.ones((9, 9), np.uint8)
        kernel_dilation = np.ones((9, 9), np.uint8)
        img_bin = cv2.erode(img_bin, kernel_erosion, iterations=2)
        img_bin = cv2.dilate(img_bin, kernel_dilation, iterations=2)

    return img_bin


def detect_contours(img_bin, min_area=0, max_area=-1, wh_ratio=2.0):
    """detect contours in a binary image.
    Args:
        img_bin: a binary image.
        min_area: the minimum area of the contours detected.
            (default: 0)
        max_area: the maximum area of the contours detected.
            (default: -1, no maximum area limitation)
        wh_ratio: the ration between the large edge and short edge.
            (default: 2.0)
    Return:
        rects: a list of rects enclosing the contours. if no contour is detected, rects=[]
    """
    rects = []
    _, contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects


def draw_rects_on_img(img, rects):
    """ draw rects on an image.
    Args:
        img: an image where the rects are drawn on.
        rects: a list of rects.
    Return:
        img_copy: an image with rects.
    """
    img_copy = img.copy()
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy


def hog_extra_and_svm_class(proposal, svm_model, resize=(64, 64)):
    """classify the region proposal.
    Args:
        proposal: region proposal (numpy array).
        svm_model: a SVM model.
        resize: resize the region proposal
            (default: (64, 64))
    Return:
        cls_prop: propabality of all classes.
    """
    img = cv2.cvtColor(proposal, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, resize)
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
    img = clahe.apply(img)
    nbins = 9  # bin number
    cell_size = (8, 8)  # number of pixels per cell
    block_size = (3, 3)  # number of cells per block

    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    features = hog.compute(img)
    features = np.reshape(features, (1, -1))
    cls_prop = svm_model.predict_proba(features)
    return cls_prop


def center_crop(img_array, crop_size=-1, resize=-1):
    """ crop and resize a square image from the centeral area.
	Args:
		img_array: image array
		crop_size: crop_size (default: -1, min(height, width)).
		resize: resized size (default: -1, keep cropped size)
	Return:
		img_crop: copped and resized image.
	"""

    rows = img_array.shape[0]
    cols = img_array.shape[1]

    if crop_size == -1 or crop_size > max(rows, cols):
        crop_size = min(rows, cols)
    row_s = max(int((rows - crop_size) / 2), 0)
    row_e = min(row_s + crop_size, rows)
    col_s = max(int((cols - crop_size) / 2), 0)
    col_e = min(col_s + crop_size, cols)

    img_crop = img_array[row_s:row_e, col_s:col_e, ]

    if resize > 0:
        img_crop = cv2.resize(img_crop, (resize, resize))
    return img_crop
