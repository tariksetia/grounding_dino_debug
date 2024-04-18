import cv2
from PIL import Image
import supervision as sv


def read_video_frames(video_path, frame_indices: list[int]):
    frames = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error Opening Stream @ {video_path}")

    frame_id = 0
    max_index = sorted(frame_indices)[-1]
    while cap.isOpened():
        _, frame = cap.read()
        if frame is None or frame_id > max_index:
            # No more frames. Recognition done.
            break

        if frame_id in frame_indices:
            frames[frame_id] = frame

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
     
    return frames

def save_annotation(image, detections, labels, f_path):
    box_annotator = sv.BoxAnnotator()
    annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
    cv2.imwrite(f_path, annotated_image)