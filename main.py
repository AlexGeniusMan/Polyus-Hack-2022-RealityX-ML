import datetime
import os.path
from os import path
import time
from pprint import pprint

import cv2
from requests import Session

from api.bboxes import *
# from .yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.predict import Predictor
from tempfile import TemporaryDirectory, NamedTemporaryFile

# MODEL = DetectMultiBackend(r"api/assets/THE_BEST.pt", device=select_device(""),
#                            dnn=False, data=None, fp16=False)


# BASE_URL = 'http://backend:8080/api/app/frame'
# BASE_PATH = "./frames"

BASE_URL = 'http://localhost:8020/api/app/frame'
BASE_PATH = "../frames"


def predict_bboxes(image_id: str, predictor: Predictor):
    st = time.time()
    preds = predictor.run(path.join(BASE_PATH, image_id), conf_thres=0.5)
    result = []
    intersection_points = []
    for bbox, conf in preds:
        x, y, w, h = rescale_bbox(bbox)
        if y < Y_MID_POINT and y + h > Y_MID_POINT:
            intersection_points.append((x, x + w))
        result.append({
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "max_mm": bbox_to_mm_max((x, y, w, h))
        })
    fullness = get_fullness(intersection_points)
    return {"bboxes": result, "fullness": fullness, "time": time.time() - st}


def process_image(image_path: str):
    image = cv2.imread(image_path)
    image = image[Y_CROP[0]:Y_CROP[1], X_CROP[0]:X_CROP[1]]
    image = cv2.fastNlMeansDenoisingColored(image, None, 7, 1, 5, 8)
    return image


if __name__ == "__main__":
    predictor = Predictor(r"./api/assets/THE_BEST.pt")
    time.sleep(10)
    requests_module = Session()
    while True:
        try:
            print(datetime.datetime.now(), 'Getting image...')
            r = requests_module.get(BASE_URL)
            image = r.json().get('image')
            image_id = str(r.json().get('id'))
            min_mm = int(r.json().get('min_mm'))
            print('min_mm', min_mm)
            # min_mm = 400
            print(datetime.datetime.now(), f"Got: {image_id}.")

            if image_id == '0':
                print(datetime.datetime.now(), 'No frames to process.')
                continue

            print(datetime.datetime.now(), 'Processing image...')
            path_to_frame = BASE_PATH + '/' + image_id + '.jpg'
            with TemporaryDirectory() as tmp_dir:
                image = process_image(path_to_frame)
                with NamedTemporaryFile(mode="w", delete=False, newline='', encoding="utf-8", dir=tmp_dir,
                                        suffix="frame.jpg") as tmp_f:
                    image_path = os.path.join(tmp_dir, tmp_f.name)
                    cv2.imwrite(image_path, image)
                    data = predict_bboxes(image_path, predictor=predictor)
            img = cv2.imread(path_to_frame)
            for bbox in data['bboxes']:
                x, y, w, h, max_mm = bbox.values()
                color = (0, 255, 0)
                if max_mm > min_mm:
                    color = (0, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.imwrite(path_to_frame, img)

            print(datetime.datetime.now(), f"Processed.")

            print(datetime.datetime.now(), 'Sending image...')
            data['id'] = image_id
            pprint(data)
            r2 = requests_module.post(BASE_URL, json=data)
            print(datetime.datetime.now(), 'Sent.')

            print('---')
        except Exception as e:
            print(e)
            raise
            print('---')
            time.sleep(3)
