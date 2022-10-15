from os import path
import time

from api.bboxes import rescale_bbox
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5 import predict


# MODEL = DetectMultiBackend(r"api/assets/THE_BEST.pt", device=select_device(""),
#                            dnn=False, data=None, fp16=False)

BASE_PATH = "./frames_to_process"

def predict_bboxes(image_id: int):
    print(select_device(""))
    st = time.time()
    preds = predict.run(r"api/assets/THE_BEST.pt", path.join(BASE_PATH, str(image_id)), conf_thres=0.5)
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
    # request
    predict_bboxes()
