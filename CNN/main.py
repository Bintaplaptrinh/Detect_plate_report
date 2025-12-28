from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import (
    IMG_PATH,
    PLATE_SEG_MODEL_PATH,
    PLATE_IMAGE_SIZE,
    CNN_MODEL_PATH,
    CNN_CLASSES_DIR,
    PLATE_WARP_SIZE,
    CHAR_DETECTION_METHOD,
    SHOW_GUI,
)
from src.utils import show
from src.models import load_cnn, load_plate_seg
from src.plate_detection import predict_plate_mask, find_plate_contour, warp_plate
from src.character_detection import detect_characters
from src.character_recognition import predict_char_with_position


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #laod img
    img_path = Path(IMG_PATH)
    if not img_path.exists():
        raise FileNotFoundError(f"Img not found: {img_path}")
    
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read img: {img_path}")
    
    print(f"Image: {img_path.name} ({img.shape[1]}x{img.shape[0]})", flush=True)
    show("01 - Input", img, scale=0.5)
    
    
    seg_model = load_plate_seg(PLATE_SEG_MODEL_PATH, device)
    mask = predict_plate_mask(seg_model, img, PLATE_IMAGE_SIZE, device)
    show("02 - Plate Mask", mask, scale=0.5)
    
    
    result = find_plate_contour(mask)
    if result is None:
        print("404 not found")
        if SHOW_GUI:
            cv2.waitKey(0)
        return
    
    cnt, box = result
    
    # draw pology on img origion
    vis = img.copy()
    cv2.polylines(vis, [box.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)
    show("03 - Detected Plate", vis, scale=0.5)
    
    #awr plate
    warped_bgr, warped_gray = warp_plate(img, box, PLATE_WARP_SIZE)
    show("04 - Warped Plate", warped_bgr, scale=2.0)
    
    char_boxes = detect_characters(warped_bgr, warped_gray, method=CHAR_DETECTION_METHOD,)
    
    #save before draw
    roi_original = warped_bgr.copy()
    roi = warped_bgr.copy()
    
    #draw box
    for (x, y, w, h) in char_boxes:
        cv2.rectangle(roi, (x, y), (x + w, y + h), (100, 255, 255), 1)
    
    
    cnn_model, cnn_classes, cnn_device = load_cnn(str(CNN_MODEL_PATH), str(CNN_CLASSES_DIR))
    
    height = warped_bgr.shape[0]
    
    first_line_chars = [(x, y, w, h) for (x, y, w, h) in char_boxes if y < height / 3]
    second_line_chars = [(x, y, w, h) for (x, y, w, h) in char_boxes if y >= height / 3]
    
    first_line = ""
    second_line = ""
    
    all_chars_sorted = []
    
    for char_box in sorted(first_line_chars, key=lambda b: b[0]):
        all_chars_sorted.append((char_box, 1)) 
    
    for char_box in sorted(second_line_chars, key=lambda b: b[0]):
        all_chars_sorted.append((char_box, 2))
    
    for position, (char_box, line_num) in enumerate(all_chars_sorted):
        x, y, w, h = char_box
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        char_img = roi_original[y:y+h, x:x+w]
        char = predict_char_with_position(cnn_model, cnn_classes, cnn_device, char_img, position)
        
        if line_num == 1:
            first_line += char
        else:
            second_line += char
        
        cv2.putText(roi, char, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    

    plate_result = f"{first_line} - {second_line}" if second_line else first_line
    print(f"result: {plate_result}", flush=True)
    
    show("05 - Result", roi, scale=2.0)
    
    #draw result
    result_img = vis.copy()
    box_int = box.astype(np.int32)
    text_x = int(box_int[:, 0].min())
    text_y = int(box_int[:, 1].min()) - 10
    cv2.putText(result_img, plate_result, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    show("06 - Final Result", result_img, scale=0.5)
    
    if SHOW_GUI:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
