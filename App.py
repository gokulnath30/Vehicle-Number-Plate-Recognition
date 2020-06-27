import cv2,pytesseract,imutils,os,re
import numpy as np

def objectDetection(input_im):
    # Load Yolo
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    classes = []
    label = None
    
    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
        
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading image
    img = imutils.resize(input_im, width=500) 
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)


    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            break
    
    return label,img


def Detect_number(new_img):
    
    image = new_img
    carNumber = None
    crop_img = 'temp/crop.png'
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 170, 200)
    cnts, new  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    img1 = image.copy()
    cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    
    
    NumberPlateCnt = None 
    img2 = image.copy()
    cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
    
    
    try: 
        os.remove(crop_img)
    except: pass

    for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:  
                NumberPlateCnt = approx 
                x, y, w, h = cv2.boundingRect(c)
                img_crop = gray[y:y + h, x:x + w] 
                cv2.imwrite(crop_img, img_crop) 
                break
           
    try:    
        cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    except: pass
     
    
    try:
        carNumber = pytesseract.image_to_string(crop_img, lang='eng')
    except:
        carNumber = '000000000'
        
    cv2.imwrite('static\img\output.png', image)
    return carNumber,image



def casecade(image):
    car_cascade = cv2.CascadeClassifier('haar_cascade\haarcascade_russian_plate_number.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    (width, height) = (300,150)    

    for (x,y,w,h) in cars:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
        car_img = gray[y:y + h, x:x + w]
        car_resize = cv2.resize(car_img, (width, height))    
        cv2.imwrite(('temp/'+'crop.png'), car_img) 

    crop_img = cv2.imread('temp/crop.png')          
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    cv2.imwrite('static\img\output.png', image) 
    
    try:
        carNumber = pytesseract.image_to_string(crop_img, lang='eng')
    except:
        carNumber = "Nill"
    
    
    return carNumber,image
    

if __name__ == "__main__":
    
    # load input iamge 
    input_im = cv2.imread("cars/7.png")
    object_vals = objectDetection(input_im)
    cv2.imwrite('temp/crops.png', input_im[1]) 
    
    if object_vals[0] == 'car' or object_vals[0] == 'truck' or object_vals[0] =='bus':
        result_vals = Detect_number(object_vals[1])
        if result_vals[0]  != '':
            print(object_vals[0],result_vals[0])
            re.sub('[^A-Za-z0-9]+', '', result_vals[0]) 
        else:
            second_check = casecade(object_vals[1])
            print(second_check[0],second_check[0])
            
            cv2.imshow("Output", second_check[1])
            cv2.waitKey(0)
    else:
        print(object_vals[0])
        
