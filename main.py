
from PIL import Image
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from groundingdino.util.inference import Model
from utils import read_video_frames, save_annotation

TEXT_PROMPT = "Faces."


class VideoRequest(BaseModel):
    video_path: str
    frames: list[int] = Field( default=[])
    
    
# @asynccontextmanager
# async def lifespan(fastapi_app: FastAPI):
#     config = read_config_deployed()
#     logger.debug(config)
    
#     detection_model = GroundingDino(
#         config_path=f"{config.mounts.model_mount}/{config.grounding_dino.config_file}",
#         checkpoint_path=f"{config.mounts.model_mount}/{config.grounding_dino.checkpoint_file}",
#         device=config.model.device,
#         text_prompt="Faces",
#         model=None
#     )
#     detection_model.load_model()
#     assert detection_model.model is not None, ValueError("detection model couldn't be loaded")
    
#     fastapi_app.detection_model = detection_model
#     fastapi_app.config = config
#     yield

# app = FastAPI(lifespan=lifespan)

model = Model("./grounding_dino/config/GroundingDINO_SwinT_OGC.py", "./grounding_dino/weights/groundingdino_swint_ogc.pth", device="cpu")


app = FastAPI()

@app.get("/video")
def detect_faces_in_video():
    frames = read_video_frames("video.mp4", [4,10,32])

    predictions = {
        k: model.predict_with_caption(v, TEXT_PROMPT)
        for k,v in frames.items()
    }

    bboxes = {
        k: v[0].xyxy.tolist()
        for k,v in predictions.items()

    }
    
    for frame_id, result in predictions.items():
        detections, labels = result
        f_path = f"bboxes/api-{frame_id}.jpg"
        save_annotation(frames[frame_id], detections, labels, f_path)
    
    return bboxes