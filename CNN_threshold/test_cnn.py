import math
import os

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

import Preprocess


ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

n = 1

Min_char = 0.01
Max_char = 0.09
img_path = r"../3.jpg"


# follow by image what test
IMG_W = 268
IMG_H = 177

# CNN settings
CNN_IMAGE_SIZE = 64
MODEL_PATH = "./cnn_model.pth"
CLASSES_DIR = r"D:/DoAnXuLiAnh/2512/archive/CNN_letter_Dataset"


def load_classes(classes_dir: str) -> list[str]:
    # Match ImageFolder class ordering
    dataset = ImageFolder(classes_dir)
    return list(dataset.classes)


def load_cnn(model_path: str, classes_dir: str) -> tuple[torch.nn.Module, list[str], torch.device]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing CNN model '{model_path}'. Run training in cnn.ipynb to create it."
        )
    if not os.path.isdir(classes_dir):
        raise FileNotFoundError(f"Missing dataset folder '{classes_dir}' for class mapping.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    # Get class names from dataset folder
    classes = load_classes(classes_dir)
    
    return model, classes, device


# Convert cropped character (binary/grayscale) to model input tensor
char_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((CNN_IMAGE_SIZE, CNN_IMAGE_SIZE)),
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.ToTensor(),
    ]
)


def predict_char_cnn(
    model: torch.nn.Module,
    classes: list[str],
    device: torch.device,
    img_roi: np.ndarray,
) -> str:
    # img_roi is expected to be a single-channel uint8 image (0..255)
    if img_roi is None or img_roi.size == 0:
        return "?"

    x = char_transform(img_roi).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        idx = int(torch.argmax(logits, dim=1).item())
    return classes[idx]


img = cv2.imread(img_path)

img_height, img_width, _ = img.shape

img = cv2.resize(img, dsize=(img_width, img_height))

cnn_model, cnn_classes, cnn_device = load_cnn(MODEL_PATH, CLASSES_DIR)

imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
kernel = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Dilation

# Draw contour and filter out the license plate
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contours có diện tích lớn nhất

screenCnt = []
for c in contours:
    peri = cv2.arcLength(c, True)  # Tính chu vi
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w / h
    if len(approx) == 4:
        screenCnt.append(approx)
        cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

if screenCnt is None:
    detected = 0
    print("No plate detected")
else:
    detected = 1

if detected == 1:

    for screenCnt in screenCnt:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

        # Find the angle of the license plate
        (x1, y1) = screenCnt[0, 0]
        (x2, y2) = screenCnt[1, 0]
        (x3, y3) = screenCnt[2, 0]
        (x4, y4) = screenCnt[3, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        sorted_array = array.sort(reverse=True, key=lambda x: x[1])
        (x1, y1) = array[0]
        (x2, y2) = array[1]
        doi = abs(y1 - y2)
        ke = abs(x1 - x2)
        angle = math.atan(doi / ke) * (180.0 / math.pi)

        # Crop out the license plate and align it to the right angle
        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)

        # Cropping
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        roi = img[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

        # Preprocessing and Character segmentation
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow(str(n + 20), thre_mor)
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ contour các kí tự trong biển số

        # Filter out characters
        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width

        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            char_area = w * h

            if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind

        # Character recognition
        char_x = sorted(char_x)
        first_line = ""
        second_line = ""

        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            imgROI = thre_mor[y : y + h, x : x + w]  # Crop the characters (binary)
            strCurrentChar = predict_char_cnn(cnn_model, cnn_classes, cnn_device, imgROI)
            cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

            if y < height / 3:  # decide 1 or 2-line license plate
                first_line = first_line + strCurrentChar
            else:
                second_line = second_line + strCurrentChar

        print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
        roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        n = n + 1

img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow("License plate", img)

cv2.waitKey(0)
