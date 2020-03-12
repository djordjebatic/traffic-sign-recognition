import os
import pickle
import shutil

from utils import *

# folder used to store detection images
if os.path.exists('detections'):
    shutil.rmtree('detections')
os.mkdir('detections')

# OPTIONS #
EQUALIZE_HSV = 0
EQUALIZE_RGB = 1

colors = ['yellow', 'red', 'blue']
'''
    Class names were labeled in serbian in dataset provided by university.

    TRANSLATION:

        OPASNOST - DANGER
        ZABRANA - FORBIDDEN
        DRUGI - OTHER
        OBAVEZNO_KRETANJE - REQUIRED_MOVEMENT

'''
cls_names = ["OPASNOST", "ZABRANA", "DRUGI", "OBAVEZNO_KRETANJE", "background"]


def train_or_load_traffic_sign_model(train_positive_images_paths, train_negative_images_path, train_image_labels):
    """
    Procedura prima listu putanja do pozitivnih i negativnih fotografija za obucavanje, liste
    labela za svaku fotografiju iz pozitivne liste, kao i putanju do foldera u koji treba sacuvati model(e) nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model(e) i da ih sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model(e) ako on nisu istranirani, ili da ih samo ucita ako su prethodno
    istrenirani i ako se nalaze u folderu za serijalizaciju

    :param train_positive_images_paths: putanje do pozitivnih fotografija za obucavanje
    :param train_negative_images_path: putanje do negativnih fotografija za obucavanje
    :param train_image_labels: labele za pozitivne fotografije iz liste putanja za obucavanje - tip znaka i tacne koordinate znaka
    :return: lista modela
    """

    models = []
    return models


def generate_bounding_boxes(rect, cols, rows, color, img_initial, clf):
    xc = int(rect[0] + rect[2] / 2)
    yc = int(rect[1] + rect[3] / 2)

    size = max(rect[2], rect[3])
    x1 = max(0, int(xc - size / 2))
    y1 = max(0, int(yc - size / 2))
    x2 = min(cols, int(xc + size / 2))
    y2 = min(rows, int(yc + size / 2))
    if color == 'yellow':
        dist_x = (x2 - x1) / 2
        x1 = x1 - dist_x
        x2 = x2 + dist_x
        x1 = int(x1)
        x2 = int(x2)
        x1 = max(0, x1)
        x2 = min(cols, x2)

        dist_y = (y2 - y1) / 2
        y1 = y1 - dist_y
        y2 = y2 + dist_y
        y1 = int(y1)
        y2 = int(y2)
        y1 = max(0, y1)
        y2 = min(rows, y2)

    proposal = img_initial[y1:y2, x1:x2]
    cls_prop = hog_extra_and_svm_class(proposal, clf)
    cls_prop = np.round(cls_prop, 2)[0]
    cls_num = np.argmax(cls_prop)
    cls_name = cls_names[cls_num]
    prop = cls_prop[cls_num]

    return x1, x2, y1, y2, cls_name, prop


def detect_traffic_signs_from_image(model_path, image_path, hsv_or_rgb):

    detections = []

    img = cv2.imread(image_path)
    img_initial = img

    if hsv_or_rgb == EQUALIZE_RGB:
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    else:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    rows, cols, _ = img.shape
    for color in colors:
        img_bin = preprocess_img(color, img, False)
        min_area = img_bin.shape[0] * img.shape[1] / (255 * 255)
        rects = detect_contours(img_bin, min_area=min_area)

        clf = pickle.load(open(model_path, 'rb'))
        img_bbx = img_initial.copy()

        for rect in rects:

            x1, x2, y1, y2, cls_name, prop = generate_bounding_boxes(rect, cols, rows, color, img_initial, clf)

            if cls_name is not "background":
                if color == "yellow":
                    cv2.rectangle(img_bbx, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_bbx, cls_name + ' ' + str(prop), (x1, y1), 1, 1.5, (0, 255, 0), 1)
                    label = [x1, y1, x2, y2, cls_name]
                else:
                    cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
                    cv2.putText(img_bbx, cls_name + ' ' + str(prop), (rect[0], rect[1]), 1, 1.5, (0, 255, 0), 1)
                    label = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3], cls_name]

                detections.append(label)
                if color == 'yellow' and cls_name != 'DRUGI':
                    detections.pop()
                if color == 'red' and cls_name == 'OBAVEZNO_KRETANJE':
                    detections.pop()
                if color == 'blue' and cls_name != 'OBAVEZNO_KRETANJE':
                    detections.pop()
                cv2.imwrite('detections/' + image_path[21:].replace('.jpg', '') + '_' + color + "_detect_result.jpg", img_bbx)

    return detections


def detect_traffic_signs(image_path, model_path='model/model_19_12'):

    print(image_path)
    detections_hsv = detect_traffic_signs_from_image(model_path, image_path, EQUALIZE_HSV)
    if len(detections_hsv) > 0:
        print(detections_hsv)
        return detections_hsv
    else:
        detections_rgb = detect_traffic_signs_from_image(model_path, image_path, EQUALIZE_RGB)
        print(detections_rgb)
        return detections_rgb
