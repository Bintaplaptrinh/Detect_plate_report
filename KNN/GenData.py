import numpy as np
import cv2
import sys


MIN_CONTOUR_AREA = 40

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def main():
    imgTrainingNumbers = cv2.imread("training_chars.png")# read in training numbers image
    #imgTrainingNumbers = cv2.resize(imgTrainingNumbers, dsize = None, fx = 0.5, fy = 0.5)
    
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY) #get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0) #blur

    imgThresh = cv2.adaptiveThreshold(
        imgBlurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    cv2.imshow("imgThresh", imgThresh) #show threshold image for refrence

    imgThreshCopy = imgThresh.copy()

    npaContours, hierarchy = cv2.findContours(
        imgThreshCopy,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )    
    
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
   

    intClassifications = []
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                        ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                        ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                        ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')] #Là mã ascii của mấy chữ này
    
    
    
    for npaContour in npaContours:          #for each contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour) #get bounding rect
            
            cv2.rectangle(
                imgTrainingNumbers,
                (intX, intY),
                (intX + intW, intY + intH),
                (0,0,255),
                2
            )
            imgROI = imgThresh[intY:intY+intH, intX:intX+intW] #crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize image, this will be more consistent for recognition and storage
            
            cv2.imshow("imgROI", imgROI)           #show cropped out char for reference
            cv2.imshow("imgROIResized", imgROIResized)   #show resized char for reference
            
            cv2.imshow("training_numbers.png", imgTrainingNumbers) #show training numbers image
            
            intChar = cv2.waitKey(0)         #get key press
            
            if intChar == 27: #if esc key was pressed
                sys.exit()         #exit program
                
            elif intChar in intValidChars: #else if the char is a valid char
                intClassifications.append(intChar)         #append valid char to list of chars
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)) #flatten image to 1d numpy array
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0) #add current flattened image to list of flattened images
            
            
    fltClassifications = np.array(intClassifications, np.float32) #convert classifications list to numpy array of type float
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1)) #reshape numpy array to 1d, necessary for call to train

    print("\n\ntraining complete\n")

    np.savetxt("classifications.txt", npaClassifications)
    np.savetxt("flattened_images.txt", npaFlattenedImages)  

    cv2.destroyAllWindows() 
    return

if __name__ == "__main__":
    main()