from matplotlib.patches import Polygon
import torch
import numpy
import imantics


def convert(target):
    type_target = type(target)
    if type_target == tuple or type_target == list:
        ret = []
        for tuple_child in target:
            temp_child = convert(tuple_child)
            ret.append(temp_child)
        return ret
    elif type_target == numpy.ndarray:
        ret = []
        for tuple_child in target.tolist():
            temp_child = convert(tuple_child)
            ret.append(temp_child)
        return ret
    elif type_target == dict:
        ret = {}
        for (key, value) in target.items():
            ret[key] = convert(value)
        return ret
    elif type_target == Polygon:
        return convert(target.xy)
    elif type_target == torch.Tensor:
        if target.dim() == 0:
            return target.item()
        ret = []
        for list_child in target:
            ret.append(convert(list_child))
        return ret
    else:
        return target


def convert_polygon(result,
                    with_bboxes=True,
                    with_polygons=True):
    ret = []
    pred_instances = result.pred_instances  # todo replace with len(instances)
    step_pred_instance = 0
    for pred_instance in pred_instances:
        ret_part = {'label': convert(pred_instance.labels.item()),
                    'score': convert(pred_instance.scores.item())}
        if with_bboxes:
            bbox = pred_instance.bboxes.tolist()
            ret_part['bbox'] = convert(bbox)[0]
        if with_polygons:
            mask = pred_instances.masks[step_pred_instance]
            mask_cpu = mask.cpu()
            mask_obj = imantics.Mask(mask_cpu)
            mask_polygons = mask_obj.polygons()
            ret_part['polygon'] = convert(mask_polygons.points)

        ret.append(ret_part)
        step_pred_instance = step_pred_instance + 1
    return ret


def bbox_to_centroid(bbox):
    """
    Convert a bounding box [x_min, y_min, x_max, y_max] to its centroid (center point).

    Args:
        bbox (list): A list of four numbers representing [x_min, y_min, x_max, y_max].

    Returns:
        dict: A dictionary containing the centroid coordinates (centroid_x, centroid_y).
    """
    x_min, y_min, x_max, y_max = bbox

    # Calculate the centroid coordinates
    centroid_x = (x_min + x_max) / 2
    centroid_y = (y_min + y_max) / 2

    return {"centroid_x": centroid_x, "centroid_y": centroid_y}


def calculate_centroid(polygon_vertices):
    # Initialize variables to store the sum of x and y coordinates
    sum_x = 0
    sum_y = 0
    # Calculate the sum of x and y coordinates
    for vertex in polygon_vertices[0]:
        x, y = vertex
        sum_x += x
        sum_y += y

    # Calculate the number of vertices
    num_vertices = len(polygon_vertices[0])

    # Calculate the centroid coordinates
    centroid_x = sum_x / num_vertices
    centroid_y = sum_y / num_vertices

    # Print the centroid coordinates
    return {"centroid_x": centroid_x, "centroid_y": centroid_y}
