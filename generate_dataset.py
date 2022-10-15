import cv2
import json
import os
import random

Y_CROP = (240, 720)
X_CROP = (200, 900)
SIZE = (X_CROP[1] - X_CROP[0], Y_CROP[1] - Y_CROP[0])

def get_frame_id(image_name: str) -> str:
    return image_name[5:-4]

def translate_bbox(bbox):
    x, y, w, h = bbox
    x = round(x - X_CROP[0], 3)
    y = round(y - Y_CROP[0], 3)
    return (x, y, w, h)

def normalize_bbox(bbox):
    x, y, w, h = bbox
    im_w, im_h = SIZE
    return ((x + w / 2) / im_w, (y + h / 2) / im_h, w / im_w, h / im_h)

def get_bbox_dict():
    bbox_dict = {}
    with open(r'./dataset/annot_local/train_annotation.json', 'r') as f:
        annotations = json.load(f)['annotations']
    for annotation in annotations:
        id = annotation['image_id']
        bboxes = bbox_dict.get(id, [])
        bbox = annotation['bbox']
        bbox = translate_bbox(bbox)
        if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > SIZE[0] or bbox[1] + bbox[3] > SIZE[1]:
            continue
        bbox = normalize_bbox(bbox)
        bboxes.append(bbox)
        bbox_dict[id] = bboxes
    return bbox_dict

def process_image(image_path: str):
    image = cv2.imread(image_path)
    image = image[Y_CROP[0]:Y_CROP[1], X_CROP[0]:X_CROP[1]]
    image = cv2.fastNlMeansDenoisingColored(image,None,7,1,5,8)
    return image

def write_image_label(id, bboxes, root_dir, path):
    image = process_image(f'dataset\\train\\frame{id}.jpg')
    cv2.imwrite(os.path.join(root_dir, 'images', path, f'{id}.png'), image)
    lines = []
    for bbox in bboxes:
        lines.append('0 ' + ' '.join([str(round(x, 5)) for x in bbox]))
    lines = '\n'.join(lines)
    with open(os.path.join(root_dir, 'labels', path, f'{id}.txt'), 'w') as file:
        file.write(lines)

def create_folders(root_path):
    for a in ['images', 'labels']:
        for b in ['train', 'test']:
            os.makedirs(os.path.join(root_path, a, b), exist_ok=True)

def create_dataset(root_path: str, split_rate: float = 0.7):
    create_folders(root_path)
    bboxes_dict = get_bbox_dict()
    ids = list(bboxes_dict.keys())
    random.shuffle(ids)
    split_index = int(len(ids) * split_rate)
    test_ids = ids[split_index:]
    for id, bboxes in bboxes_dict.items():
        path = 'train'
        if id in test_ids:
            path = 'test'
        write_image_label(id, bboxes, root_path, path)

if __name__ == '__main__':
    create_dataset('yolo_dataset')
    os.mkdir('public')
    for img_path in os.listdir(r'./dataset/public/'):
        image = process_image(os.path.join(r'./dataset/public/', img_path))
        cv2.imwrite(os.path.join(r'./public/', img_path), image)
