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
                "https://i.postimg.cc/BZzHPrL5/Limit-4bff52ca-7a12-11ec-99d4-a44cc849d7a3.jpg",
                "https://i.postimg.cc/xTbHv6Mw/Limit-4c0bf458-7a12-11ec-b25b-a44cc849d7a3.jpg",
                "https://i.postimg.cc/RZ5tXwTN/Limit-4c1c25ac-7a12-11ec-8acd-a44cc849d7a3.jpg"
            ]

        )
        self.loaded = True
