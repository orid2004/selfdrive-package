import os
from .base import detection_model_path, Model
from .exceptions import EmptyModel

VERSION = 3
NAME = "speedlimit"
LABELMAP_NAME = "labelmap.pbtxt"
MAX_DETECTIONS = 8
_model: Model


def load():
    global _model
    path = detection_model_path(NAME, VERSION)
    _model = Model(
        ckpt_path=path,
        label_map_path=os.path.join(path, LABELMAP_NAME),
        max_detections=MAX_DETECTIONS
    )
    print("Speedlimit model is ready to detect...")


def valid_model(func):
    global _model

    def inner(*args):
        global _model
        if not _model:
            raise EmptyModel
        return func(*args)

    return inner


@valid_model
def set_num_detections(n):
    global _model
    if n >= 0:
        _model.num_detections = n
    else:
        raise ValueError


@valid_model
def add_dataset(dataset, index=0):
    global _model
    _model.datasets.insert(index, dataset)


@valid_model
def len_datasets() -> int:
    global _model
    return len(_model.datasets)


@valid_model
def del_old_datasets():
    global _model
    _model.datasets = _model.datasets[100:]


@valid_model
def del_all_datasets():
    global _model
    _model.datasets.clear()


@valid_model
def next_dataset():
    global _model
    return _model.datasets.pop(0)


@valid_model
def allow_detections():
    global _model
    return _model.num_detections <= _model.max_detections


@valid_model
def detect(im) -> tuple:
    global _model
    tf_detections = _model.get_tf_detections(im)
    detections = _model.get_detections(im, tf_detections)
    _model.num_detections += len(detections.keys())
    return detections, tf_detections


@valid_model
def im_with_detections(im, tf_detections=None):
    global _model
    return _model.get_image_np_with_detections(im, tf_detections)
