from pathlib import Path


IMG_PATH = Path(r"../3.jpg")
# IMG_PATH = Path(r"../carlong_0004.png")


PLATE_SEG_MODEL_PATH = Path("./plate_seg.pth")
PLATE_IMAGE_SIZE = 256


CNN_MODEL_PATH = Path("./cnn_model.pth")
CNN_CLASSES_DIR = Path(r"./CNN_letter_Dataset")
CNN_IMAGE_SIZE = 64


MIN_CHAR_AREA_RATIO = 0.01
MAX_CHAR_AREA_RATIO = 0.09


PLATE_WARP_SIZE = (320, 96)


CHAR_DETECTION_METHOD = "auto"


SHOW_GUI = True