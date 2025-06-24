import os
import json
import mmcv
import pandas as pd
import config
from mmengine import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules
from common import convert_polygon, calculate_centroid, bbox_to_centroid


def process_images(list_of_images, base_path, model, file_name):
    labels = {
        0: 'Arms',
        1: 'Rings'
    }

    final_results = []
    centroid_data = []
    i = 0

    for image in list_of_images:
        if image == '_annotations.coco.json':
            continue

        img_path = os.path.join(base_path, image)
        img = mmcv.imread(img_path, channel_order='rgb')
        new_result = inference_detector(model, img)

        image_result = {'image_ame': image}
        detections = []

        processed_results = convert_polygon(new_result, with_bboxes=True, with_polygons=True)

        for result in processed_results:
            score = result['score']
            if score >= 0:
                label_name = labels[result['label']]
                bbox = result['bbox']
                polygon = calculate_centroid(result['polygon']) if len(result['polygon']) > 0 else []
                centroid = bbox_to_centroid(bbox) if bbox else []

                centroid_data.append({
                    'image': image,
                    'label': label_name,
                    'bbox': bbox,
                    'polygon': polygon,
                    'centroid': centroid
                })

                detections.append({
                    'class_label': label_name,
                    'predict_accuracy': score,
                    'bbox': bbox,
                    'centroid': centroid
                })

        image_result['data'] = detections
        final_results.append(image_result)

        i += 1
        print(image_result)
        print(i)

        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.add_datasample(
            name='result',
            image=img,
            data_sample=new_result,
            draw_gt=False,
            wait_time=0,
            out_file=f'images/{image}'
        )

    result_path = './results'
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, f'result {file_name}.json'), 'w') as fp:
        json.dump(final_results, fp)

    df = pd.DataFrame(centroid_data)
    df.to_excel(os.path.join(result_path, f'centroid_data {file_name}.xlsx'), index=False)


cfg = Config.fromfile('./config.py')
runner = Runner.from_cfg(cfg)
checkpoint_file = f'{runner.work_dir}/epoch_28.pth'
model = init_detector(cfg, checkpoint_file, device='cuda')
file_name = 'final_results'
base_path = './frames/Paper 4_Frames'
list_of_images = sorted(os.listdir(base_path))
register_all_modules()

process_images(list_of_images, base_path, model, file_name)
