import json
from groundingdino.util.inference import Model
from utils import read_video_frames


TEXT_PROMPT = "Face"

model = Model("./grounding_dino/config/GroundingDINO_SwinT_OGC.py", "./grounding_dino/weights/groundingdino_swint_ogc.pth", device="cpu")


frames = read_video_frames("video.mp4", [4,10,32])

predictions = {
    k: model.predict_with_caption(v, TEXT_PROMPT)
    for k,v in frames.items()
}

bboxes = {
    k: v[0].xyxy.tolist()
    for k,v in predictions.items()
}

with open("script_result.json", 'w') as f:
    json.dump(bboxes, f, indent=2)