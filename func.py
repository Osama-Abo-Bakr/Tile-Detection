import numpy as np
import supervision as sv


def masks_to_bool(masks):
    if type(masks) == np.ndarray:
        return masks.astype(bool)
    return masks.cpu().numpy().astype(bool)


def annotate_image(image_path: np.ndarray, masks: np.ndarray) -> np.ndarray:
    image = image_path

    xyxy = sv.mask_to_xyxy(masks=masks)
    detections = sv.Detections(xyxy=xyxy, mask=masks)

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    return mask_annotator.annotate(scene=image.copy(), detections=detections)