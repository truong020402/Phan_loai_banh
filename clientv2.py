import cv2

import socket
import struct
import time
import pickle
import serial
from picamera2 import Picamera2



ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
ser.reset_input_buffer()




img_counter = 0
#cài đặt camera và thông số nén
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 640)}))
picam2.start()
time.sleep(2)

#cài đặt socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.2.5', 8485))
connection = client_socket.makefile('wb')
my_dict = {'banh_hoa': '1', 'banh_xoay': '2', 'banh_lo': '3', 'banh_hoa_hong' : '4', 'banh_lo_hong' :'5', 'banh_xoay_hong': '6'}
while True:
    try:
        #doc tu serial   
        line = ser.readline().decode('utf-8').strip()
        if line == '1' :
            time.sleep(2)
            start_time = time.time()
            frame = picam2.capture_array()
            result, frame = cv2.imencode('.jpg', frame, encode_param)
        #    data = zlib.compress(pickle.dumps(frame, 0))
            data = pickle.dumps(frame, 0)
            size = len(data)


            print("{}: {}".format(img_counter, size))
            client_socket.sendall(struct.pack(">L", size) + data)
            img_counter += 1
            
            
            end_time = time.time()


            # Đọc dữ liệu từ socket
            # Giải mã dữ liệu thành chuỗi
            dt = ""
            message = ""
            while True:
                data = client_socket.recv(1024)
                message = data.decode()
            # In ra chuỗi đã đọc được từ socket
                print(message)
                if message == 'n':
                    break
                dt = message

            #In ra thời gian kết thúc hàm
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            #if img_counter == 1:
                #ser.write('5'.encode())
            if img_counter > 0:
                try:
                    if int(my_dict[dt]) < 4 :
                        print("gui ma sang arduino")
                        ser.write((my_dict[dt]).encode())
                        time.sleep(6)
                    else:
                        print("gui ma sang arduio")
                        ser.write('4'.encode())
                        time.sleep(6)
                except Exception:
                    print("gui ma sang arduio")
                    ser.write('4'.encode())
                    time.sleep(6)
                    print("error!")
            
            #time.sleep(5)
    except Exception:
    # Dừng vòng lặp nếu người dùng nhấn Ctrl+C
        pass

    finally:
        # Đóng kết nối Serial
        ser.close()





