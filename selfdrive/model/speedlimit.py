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
            max_detections=MAX_DETECTIONS,
            warmup_set=[
                "https://i.ibb.co/tZ4VQPS/Limit-4bff52ca-7a12-11ec-99d4-a44cc849d7a3.jpg",
                "https://i.ibb.co/y6Q7m1x/Limit-4c6a323a-7a12-11ec-83a9-a44cc849d7a3.jpg",
                "https://i.ibb.co/znSCPW1/Limit-4c5ee9ca-7a12-11ec-8e7e-a44cc849d7a3.jpg",
                "https://i.ibb.co/X3s4PsV/Limit-4c1eba6c-7a12-11ec-a633-a44cc849d7a3.jpg"
            ]

        )
        self.loaded = True
