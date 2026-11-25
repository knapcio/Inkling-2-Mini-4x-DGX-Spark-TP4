"""CPU stand-in of the sampling-accept graph holder."""


class DsparkSamplingAcceptGraph:
    def __init__(self, *, epilogue=None, **_):
        self.epilogue = epilogue
