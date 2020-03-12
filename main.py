from process import detect_traffic_signs
import glob
import sys
import os

# ------------------------------------------------------------------
if len(sys.argv) > 1:
    VALIDATION_DATASET_PATH = sys.argv[2]
else:
    VALIDATION_DATASET_PATH = '.' + os.path.sep + 'dataset' + os.path.sep + 'validation' + os.path.sep
# -------------------------------------------------------------------
# izvrsiti detekciju i klasifikaciju znakova iz validacionog skupa
processed_image_names = []
detected_traffic_signs = []

for image_path in glob.glob(VALIDATION_DATASET_PATH + "*.jpg"):
    image_directory, image_name = os.path.split(image_path)
    processed_image_names.append(image_name)
    detected_traffic_signs.append(detect_traffic_signs(image_path))

# -----------------------------------------------------------------
# Kreiranje fajla sa rezultatima detekcije za svaku sliku
result_file_contents = "image_name;x_min;y_min;x_max;y_max;sign_type\n"
for image_index, image_name in enumerate(processed_image_names):
    detection_results = detected_traffic_signs[image_index]
    for result in detection_results:
        x_min = int(result[0])
        y_min = int(result[1])
        x_max = int(result[2])
        y_max = int(result[3])
        sign_type = result[4]
        result_file_contents += "%s;%s;%s;%s;%s;%s\n" % (image_name, x_min, y_min, x_max, y_max, sign_type)

# sacuvaj formirane rezultate u csv fajl
with open('result.csv', 'w') as output_file:
    output_file.write(result_file_contents)

# ------------------------------------------------------------------
