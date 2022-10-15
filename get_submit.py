from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import os
import cv2

Y_CROP = (240, 720)
X_CROP = (200, 900)
SIZE = (X_CROP[1] - X_CROP[0], Y_CROP[1] - Y_CROP[0])
WRITE_FILES = False
PATH = r'./yolov5/runs/detect/exp2/labels'


def get_frame_id(image_name: str) -> str:
    return image_name[5:-4]


def rescale_bbox(bbox):
    x, y, w, h = bbox
    im_w, im_h = SIZE
    x = int(round((x - w / 2) * im_w + X_CROP[0]))
    y = int(round((y - h / 2) * im_h + Y_CROP[0]))
    w = int(round(w * im_w))
    h = int(round(h * im_h))
    return (x, y, w, h)


coco = Coco()
coco.add_category(CocoCategory(id=0, name='stone0'))
coco.add_category(CocoCategory(id=1, name='stone1'))
bboxes = os.listdir(PATH)
for bbox_filepath in bboxes:
    id = get_frame_id(bbox_filepath)
    coco_image = CocoImage(file_name=f'./dataset/public/frame{id}.jpg',
                           height=720, width=1280, id=id)
    if WRITE_FILES:
        image = cv2.imread(f'./dataset/public/frame{id}.jpg')

    with open(os.path.join(PATH, bbox_filepath)) as f:
        for line in f:
            bbox = [float(x.strip()) for x in line.split(' ')[1:]]
            bbox = rescale_bbox(bbox)
            x, y, w, h = bbox
            if WRITE_FILES:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            coco_image.add_annotation(
                                CocoAnnotation(
                                    bbox=bbox,
                                    category_id=1,
                                    category_name='stone1',
                                    image_id=id
                                )
                            )
    if WRITE_FILES:
        cv2.imwrite(f'tmp/bboxes/frame{id}.jpg', image)
    coco.add_image(coco_image)
save_json(data=coco.json, save_path=r'./submit.json')
