import datetime
from os import path
import time
from pprint import pprint
from requests import Session

from api.bboxes import *
# from .yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.predict import Predictor

# MODEL = DetectMultiBackend(r"api/assets/THE_BEST.pt", device=select_device(""),
#                            dnn=False, data=None, fp16=False)


BASE_URL = 'http://backend:8080/api/app/frame'
BASE_PATH = "./frames"


def predict_bboxes(image_id: str, predictor: Predictor):
    print(select_device(""))
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


if __name__ == "__main__":
    requests_module = Session()
    predictor = Predictor(r"./api/assets/THE_BEST.pt")
    while True:
        try:
            print(datetime.datetime.now(), 'Getting image...')
            r = requests_module.get(BASE_URL)
            image = r.json().get('image')
            image_id = str(r.json().get('id'))
            # image_id = 'test.jpg'
            print(datetime.datetime.now(), f"Got: {image_id}.")

            if image_id == '0':
                print(datetime.datetime.now(), 'No frames to process.')
                continue

            print(datetime.datetime.now(), 'Processing image...')
            data = predict_bboxes(image_id + '.jpg', predictor=predictor)
            print(datetime.datetime.now(), f"Processed.")

            print(datetime.datetime.now(), 'Sending image...')
            data['id'] = image_id
            pprint(data)
            r2 = requests_module.post(BASE_URL, data=data)
            print(datetime.datetime.now(), 'Sent.')

            print('---')
        except Exception as e:
            print(e)
            print('---')
            time.sleep(3)
