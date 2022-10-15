import datetime
from os import path
import time
from pprint import pprint
from requests import Session

from api.bboxes import rescale_bbox
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5 import predict

# MODEL = DetectMultiBackend(r"api/assets/THE_BEST.pt", device=select_device(""),
#                            dnn=False, data=None, fp16=False)


BASE_URL = 'http://localhost:8020/api/app/frame'
BASE_PATH = "../frames"


def predict_bboxes(image_id: str):
    print(select_device(""))
    st = time.time()
    preds = predict.run(r"../api/assets/THE_BEST.pt", path.join(BASE_PATH, image_id), conf_thres=0.5)
    result = []
    for bbox, conf in preds:
        x, y, w, h = rescale_bbox(bbox)
        result.append({
            "x": x,
            "y": y,
            "width": w,
            "height": h,
        })
    return {"bboxes": result, "time": time.time() - st}


if __name__ == "__main__":
    requests_module = Session()
    print(datetime.datetime.now(), 'Getting image...')
    r = requests_module.get(BASE_URL)
    image = r.json().get('image')
    # image_id = str(r.json().get('id'))
    image_id = 'test.jpg'
    print(datetime.datetime.now(), f"Got: {image_id}.")

    print(datetime.datetime.now(), 'Processing image...')
    data = predict_bboxes(image_id)
    print(datetime.datetime.now(), f"Processed.")

    print(datetime.datetime.now(), 'Sending image...')
    data['id'] = image_id
    pprint(data)
    # r2 = requests_module.post(BASE_URL, data=data)
    print(datetime.datetime.now(), 'Sent.')

    print('---')
