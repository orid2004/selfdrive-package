import keras_ocr

pipeline: keras_ocr.pipeline.Pipeline


def load_self():
    global pipeline
    pipeline = keras_ocr.pipeline.Pipeline()
    warmup_images = [
        'https://storage.googleapis.com/gcptutorials.com/examples/keras-ocr-img-1.jpg',
        'https://storage.googleapis.com/gcptutorials.com/examples/keras-ocr-img-2.png',
        r'C:\Users\Ori\PycharmProjects\packages\selfdrive\selfdrive\test\Limit-4c1eba6c-7a12-11ec-a633-a44cc849d7a3.jpg',
        r'C:\Users\Ori\PycharmProjects\packages\selfdrive\selfdrive\test\Limit-4c3e8fa8-7a12-11ec-a8fe-a44cc849d7a3.jpg'
    ]
    predict(warmup_images)


def predict(input_images):
    images = [
        keras_ocr.tools.read(url) for url in input_images
    ]
    prediction_groups = pipeline.recognize(images)
    for i in range(len(prediction_groups)):
        predicted_image = prediction_groups[i]
        for text, box in predicted_image:
            return text