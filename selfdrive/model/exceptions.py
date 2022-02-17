class EmptyModel(Exception):
    def __init__(self):
        super().__init__("An attempt to access an empty model. Load it first.")