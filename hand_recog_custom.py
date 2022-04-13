import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time
import platform
from PIL import ImageFont, ImageDraw, Image


def cv2_draw_label(image, text, point):
    x, y = point
    x, y = int(x), int(y)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    if platform.system() == 'Darwin':
        font = 'AppleGothic.ttf'
    elif platform.system() == 'Windows': 
        font = 'malgun.ttf'
    elif platform.system() == 'Linux': 
        font = 'malgun.ttf'
    try:
        imageFont = ImageFont.truetype(font, 28)
    except:
        imageFont = ImageFont.load_default()
    draw.text((x, y), text, font=imageFont, fill=(255, 255, 255))
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image

max_num_hands = 1

# 34개? backspace랑 space까지, 쌍자음은 어떻게?
gesture = {
    0:'ㄱ', 2:'ㄴ', 3:'ㄷ', 4:'ㄹ', 5:'ㅁ', 6:'ㅂ', 7:'ㅅ', 8:'ㅇ', 9:'ㅈ', 10:'ㅊ', 11:'ㅋ', 12:'ㅌ', 13:'ㅍ', 14:'ㅎ',
    15:'ㅏ', 16:'ㅑ', 17:'ㅓ', 18:'ㅕ', 19:'ㅗ', 20:'ㅛ', 21:'ㅜ', 22:'ㅠ', 23:'ㅡ', 24:'ㅣ', 25:'ㅐ', 26:'ㅔ', 27:'ㅚ',
    28:'ㅟ', 29:'ㅒ', 30:'ㅖ', 31:'ㅢ', 32:'SPACE', 33:'BACKSPACE'
}

# gesture = {
#     0:'ㄱ', 1:'ㄴ', 2:'ㄷ', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k',
#     11:'l', 12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u',
#     21:'v', 22:'w', 23:'ㅁ', 24:'ㅂ', 25:'ㅅ', 26:'spacing'
# }

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

f = open('test.txt', 'w')

file = np.genfromtxt('dataSet.txt', delimiter=',')
angleFile = file[:,:-1]
labelFile = file[:,-1]
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)
knn = cv2.ml.KNearest_create()

knn.train(angle, cv2.ml.ROW_SAMPLE, label)
cap = cv2.VideoCapture(0)

startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 2

while True:
    ret, img = cap.read()
    if not ret:
        continue
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,3))
            for j,lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]
            
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0,  9, 10, 11,  0, 13, 14, 15,  0, 17, 18, 19],:]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],:]
            
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            compareV1 = v[[0, 1, 2, 4, 5, 6, 7,  8,  9, 10, 12, 13, 14, 16, 17],:]
            compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],:]

            angle = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))
            angle = np.degrees(angle)
            
            if keyboard.is_pressed('a'):
                for num in angle:
                    num = round(num, 6)
                    f.write(str(num))
                    f.write(',')
                f.write("0")
                f.write('\n')
                print("next")
            
            # data = np.array([angle], dtype=np.float32)
            # ret, results, neighbours, dist = knn.findNearest(data, 3)
            # index = int(results[0][0])
            # if index in gesture.keys():
            #     if index != prev_index:
            #         startTime = time.time()
            #         prev_index = index
            #     else:
            #         if time.time() - startTime > recognizeDelay:
            #             if index == 26:
            #                 sentence += ' '
            #             else:
            #                 sentence += gesture[index]
            #             startTime = time.time()
                
                # cv2.putText(img, gesture[index].upper(), (int(res.landmark[0].x * img.shape[1] - 10),
                #             int(res.landmark[0].y * img.shape[0] + 40)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255,255),3)
                
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
        
        
        # img = cv2_draw_label(img, sentence, (30, 30))
        
        cv2.imshow('HAND', img)
        cv2.waitKey(1)
        if keyboard.is_pressed('q'):
            break
f.close()
        