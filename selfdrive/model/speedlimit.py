import os
from .base import Model as Base
from .base import detection_model_path

VERSION = 3
NAME = "speedlimit"
LABELMAP_NAME = "labelmap.pbtxt"
MAX_DETECTIONS = 10


class Model(Base):
    def __init__(self):
        self.path = detection_model_path(NAME, VERSION)
        self.loaded = False

    def load_self(self):
        super().__init__(
            ckpt_path=self.path,
            label_map_path=os.path.join(self.path, LABELMAP_NAME),
            max_detections=MAX_DETECTIONS

        )
        self.loaded = True
