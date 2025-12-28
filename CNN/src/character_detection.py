import cv2
import numpy as np


#mser
def detect_characters_mser(
    gray_img: np.ndarray,
    min_area: int = 50,
    max_area: int = 3000,
    min_aspect: float = 0.15,
    max_aspect: float = 0.9,
) -> list[tuple[int, int, int, int]]:


    mser = cv2.MSER_create()
    mser.setDelta(5)
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    mser.setMaxVariation(0.25)
    mser.setMinDiversity(0.2)
    

    regions, _ = mser.detectRegions(gray_img)
    
    img_height, img_width = gray_img.shape[:2]
    

    raw_boxes = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        

        min_height = img_height * 0.15
        max_height = img_height * 0.85
        
        if min_height < h < max_height and h > 0:
            aspect = w / h
            if min_aspect < aspect < max_aspect:
                area = w * h
                if min_area < area < max_area:
                    raw_boxes.append((x, y, w, h))
    
    boxes = _nms_boxes(raw_boxes, overlap_thresh=0.3)
    

    if len(boxes) > 0:
        heights = [b[3] for b in boxes]
        median_height = sorted(heights)[len(heights) // 2]
        
        boxes = [
            (x, y, w, h) for (x, y, w, h) in boxes
            if 0.5 * median_height < h < 1.5 * median_height
        ]
    

    boxes = sorted(boxes, key=lambda b: b[0])
    
    return boxes


#contour
def detect_characters_contour(bgr_img: np.ndarray, gray_img: np.ndarray = None, min_height_ratio: float = 0.25, max_height_ratio: float = 0.95, min_aspect: float = 0.15, max_aspect: float = 0.9,) -> list[tuple[int, int, int, int]]:

    if bgr_img is None:
        return []
    

    if gray_img is None:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_img.copy()
    
    img_height, img_width = gray.shape[:2]
    

    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    raw_boxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        height_ratio = h / img_height
        aspect_ratio = w / h if h > 0 else 0
        
        if min_height_ratio < height_ratio < max_height_ratio:
            
            if min_aspect < aspect_ratio < max_aspect:
                area = w * h
                min_area = img_width * img_height * 0.005
                max_area = img_width * img_height * 0.25
                
                if min_area < area < max_area:
                    raw_boxes.append((x, y, w, h))
    
    boxes = _nms_boxes(raw_boxes, overlap_thresh=0.3)
    
    if len(boxes) >= 3:
        heights = [b[3] for b in boxes]
        median_height = sorted(heights)[len(heights) // 2]
        
        boxes = [
            (x, y, w, h) for (x, y, w, h) in boxes
            if 0.5 * median_height < h < 1.8 * median_height
        ]
    
    boxes = sorted(boxes, key=lambda b: b[0])
    
    return boxes


#auto choose
def detect_characters_auto(bgr_img: np.ndarray, gray_img: np.ndarray, min_expected_chars: int = 4, max_expected_chars: int = 12,) -> list[tuple[int, int, int, int]]:

    mser_boxes = detect_characters_mser(gray_img)
    mser_count = len(mser_boxes)
    

    contour_boxes = detect_characters_contour(bgr_img, gray_img)
    contour_count = len(contour_boxes)
    

    def score_result(boxes, count):
        if count == 0:
            return -1
        
        if min_expected_chars <= count <= max_expected_chars:
            base_score = 100
        else:
            base_score = 50 - abs(count - (min_expected_chars + max_expected_chars) / 2) * 5
        
        if len(boxes) >= 2:
            heights = [b[3] for b in boxes]
            height_variance = np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 1
            uniformity_score = max(0, 30 * (1 - height_variance))
            base_score += uniformity_score
        
        return base_score
    
    mser_score = score_result(mser_boxes, mser_count)
    contour_score = score_result(contour_boxes, contour_count)
    
    if mser_score >= contour_score:
        return mser_boxes
    else:
        return contour_boxes


#detect characters
def detect_characters(bgr_img: np.ndarray, gray_img: np.ndarray, method: str = "auto",) -> list[tuple[int, int, int, int]]:

    method = method.lower().strip()
    
    if method == "mser":
        return detect_characters_mser(gray_img)
    elif method in ("segment", "contour"):
        return detect_characters_contour(bgr_img, gray_img)
    elif method == "auto":
        return detect_characters_auto(bgr_img, gray_img)
    else:

        print(f"unknown method '{method}', using 'auto'")
        return detect_characters_auto(bgr_img, gray_img)



def _nms_boxes(boxes: list[tuple[int, int, int, int]], overlap_thresh: float = 0.3,) -> list[tuple[int, int, int, int]]:

    if len(boxes) == 0:
        return []
    
    boxes_arr = np.array(boxes)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
    y2 = boxes_arr[:, 1] + boxes_arr[:, 3]
    areas = boxes_arr[:, 2] * boxes_arr[:, 3]
    
    idxs = np.argsort(x1)
    
    picked = []
    while len(idxs) > 0:
        i = idxs[0]
        picked.append(i)
        
        if len(idxs) == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[idxs[1:]] - intersection
        iou = intersection / (union + 1e-6)
        
        remaining = np.where(iou <= overlap_thresh)[0]
        idxs = idxs[remaining + 1]
    
    return [boxes[i] for i in picked]


def sort_contours(cnts):

    if len(cnts) == 0:
        return []
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][0], reverse=False))
    return cnts


def segment_characters(plate_img):

    if plate_img is None:
        return None, []

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0)
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_list = []
    
    vis_plate = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if len(cnts) > 0:
        cnts = sort_contours(cnts)
        
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            
            ratio = h / w
            height_ratio = h / gray.shape[0]

            if 0.4 < height_ratio < 0.95 and ratio > 0.3:

                char_roi = thresh[y:y+h, x:x+w]

                char_roi = cv2.resize(char_roi, (30, 60))

                char_list.append(char_roi)
                
                cv2.rectangle(vis_plate, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return vis_plate, char_list