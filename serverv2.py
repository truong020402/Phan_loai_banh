import socket
import cv2
import pickle
import numpy as np
import struct ## new
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.models import  load_model
import sys
from ultralytics import YOLO
import math
from keras.optimizers import Adam

class_name = ['banh_hoa', 'banh_hoa_hong',
'banh_lo', 'banh_lo_hong',
'banh_xoay' , 'banh_xoay_hong'
]

dic_class_name = {'banh_hoa': '1', 'banh_lo': '2', 'banh_xoay': '3',
                  'banh_hoa_hong': '4', 'banh_lo_hong': '5', 'banh_xoay_hong': '6'}

def get_model():
    vgg_model = VGG16(weights='imagenet',
                    include_top=False, 
                    input_shape=(224, 224, 3))

    for layer in vgg_model.layers[:15]:
        layer.trainable = False

    x = vgg_model.output
    x = Flatten()(x) # Flatten dimensions to for use in FC layers
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x= Dropout(0.4)(x)
    x = Dense(6, activation='softmax')(x) # Softmax for multiclass
    transfer_model = Model(inputs=vgg_model.input, outputs=x)
    learning_rate= 0.0001
    transfer_model.compile(loss="categorical_crossentropy",
                        optimizer= Adam(lr=learning_rate),
                        metrics=["accuracy"])

    return transfer_model

# Load weights model da train
my_model = get_model()
my_model.load_weights(r"D:\\HK6\\PBL5\\TCP\\vgg16FineTuning2.h5")
print(my_model.summary())

def myDetect(my_model, image):
        class_name = ['banh_hoa', 'banh_hoa_hong',
                    'banh_lo', 'banh_lo_hong',
                    'banh_xoay' , 'banh_xoay_hong'
                    ]
        
        # Resize
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # áp dụng phương pháp Adaptive Thresholding để chuyển đổi sang ảnh đen trắng
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

            # Decrease brightness by reducing value channel in HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.subtract(v, 30)
                    
            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

            
            image = cv2.resize(img, dsize=(224, 224), interpolation = cv2.INTER_AREA)
            image = image.astype('float')*1./255
            # Convert to tensor
            image = np.expand_dims(image, axis=0)

            # Predict
            predict = my_model.predict(image)
            print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
            print(np.max(predict[0],axis=0))
            if np.max(predict) >= 0.7:
                return class_name[np.argmax(predict[0])]
            else:
                return ""
        except Exception as e:
            return ""    
            
            


# Load the YOLOv8 model
model = YOLO('D:\\HK6\\PBL5\\TCP\\best.pt')



while True:
    HOST=''
    PORT=8485
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')
    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(1)
    print('Socket now listening')

    conn,addr=s.accept()

    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))
    while True:
        try:
            if conn.fileno() == -1:
                continue
            while len(data) < payload_size:
                print("Recv: {}".format(len(data)))
                dt = conn.recv(4096)
                if not dt:
                    print("CLient mất kết nối!")
                    break
                data += dt

            print("Done Recv: {}".format(len(data)))
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            print("msg_size: {}".format(msg_size))
            while len(data) < msg_size:
                data += conn.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            
            results = model(frame, conf=0.6, show_labels = False, show_conf = True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            boxes = results[0].boxes
            i = 0
            my_detect = "" 
            try:
                x = 4000
                y = 0
                w = 0
                h = 0
                for box in boxes:
                    # returns one box
                    print(box.xyxy)
                    x1,y1,w1,h1 = box.xyxy.flatten().tolist()
                    print(x1,y1,w1,h1)  
                    if x1 < x:
                        x,y,w,h = x1,y1,w1,h1  

                left   = math.ceil(x)
                top    = math.ceil(y)
                right  = math.ceil(w)
                bottom = math.ceil(h)
                if abs(1 - (left - right)/(top - bottom)) > 0.2:
                    continue 
                        # Cắt ảnh theo tọa độ
                object_img = frame[top:bottom, left:right]
                    #cv2.imshow("cut",object_img)
                        #myDetect(my_model=my_model, img = object_img)        
                my_detect = myDetect(my_model=my_model, image = object_img)
                color = (0, 255, 0)        
                cv2.putText( annotated_frame, my_detect , (left - 10, bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    #trả kết quả detected qua sockets
                if my_detect != "":
                    conn.sendall(my_detect.encode())
                    print (my_detect)
            except Exception:
                print("next frame is error")    
                conn.sendall('#'.encode())                
                
            finally:
                print("next frame is error")    
                conn.sendall('#'.encode()) 
                
            print("next frame")    
            conn.sendall('#'.encode())
                
            # Display the annotated frame
            
            cv2.imshow("object detection", annotated_frame)
                
                # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception:
            print("Clients ngắt kết nối")
            break
        
