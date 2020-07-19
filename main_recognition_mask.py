# -*- coding: utf-8 -*-
"""
Created on Fri May 15 08:57:37 2020

@author: ChingyingHuang
"""

#-------------------------------------------------------------------------
from PIL import Image, ImageTk,ImageDraw, ImageFont
import tkinter as tk
import argparse
import datetime
import cv2
import os
import php
#import sys
#from skimage import io
import dlib
import numpy
#import msvcrt

#import imutils
#import time
from keras.models import load_model
import pyttsx3
#import face_recognition

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time



my = php.kit()
engine = pyttsx3.init()
startTime = datetime.datetime.now()

# OpenCV人脸识别分类器---------------
#classifier = cv2.CascadeClassifier(
#    "haarcascade_frontalface_default.xml"
#)
#----------------------------------
emotion_classifier = load_model(
        'classifier/emotion_models/simple_CNN.530-0.65.hdf5')

np_dir = "./rec/result"
#faces_numpy_folder_path = "./rec/有口罩"
faces_numpy_folder_path = "./out5_14"#rec/numpy_ch"

facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")# 載入人臉辨識檢測器 # 人臉辨識模型
detector = dlib.get_frontal_face_detector()# 載入人臉檢測器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')# 載入人臉特徵點檢測器 人臉68特徵點模型

descriptors = []
candidate = []


for f in my.glob(faces_numpy_folder_path+"\\*.npy"):
    base = os.path.basename(f)
    candidate.append(os.path.splitext(base)[0])
    v = numpy.load(f)
    descriptors.append(v)
#cap = cv2.VideoCapture(0)
#fps = cap.get(cv2.CAP_PROP_FPS)
#print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
#
#cap.set(cv2. CAP_PROP_FRAME_WIDTH, 320)
#cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 240)
    
#faceNet-----------------------------------------------------------
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
#time.sleep(2.0)
#-----------------------------------------------------------

fp = open("filename.txt", "a")

class Application:
    
    def __init__(self, output_path = "./"):        
        self.step_while = 0;
        """ Initialize application which uses OpenCV + Tkinter. It displays
        a video stream in a Tkinter window and stores current snapshot on disk """
        self.vs = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera
        #       # 获取视频的一些参数, 这里是帧速率
        #       fps = self.vs.get(5)
        #       # 获取视频的总帧数
        #       x = self.vs.get(7)
        #       print(fps)
        #       print(x)
           
        #       self.vs.set(cv2. CAP_PROP_FRAME_WIDTH, 320)#320
        #       self.vs.set(cv2. CAP_PROP_FRAME_HEIGHT, 240)#240
        self.output_path = output_path  # store output path
        self.current_image = None  # current image from the camera
        #       self.root = tk.Toplevel()
        self.root = tk.Tk()  # initialize root window
        self.root.title("Face recognition")  # set window title
        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
           
        self.x1=0
        self.last_rec_name=""
        self.l = tk.Label(self.root, bg='Thistle', width=20, text='人臉心情辨識打卡系統').pack()      
        
        btn = tk.Button(self.root, width = 20, text="Make up!", command=self.take_makeup)
        btn.pack()#,side='bottom'        fill="both", expand=True, padx=10, pady=10
        
        self.rootface_rec_signin = tk.IntVar()
        self.rootface_rec_signin.set(0)       # 預設值0=未報到
       
        face_rec = tk.Checkbutton(self.root, text='人臉辨識', variable=self.rootface_rec_signin,onvalue=1, offvalue=0)
        self.rootface_rec_signin = tk.BooleanVar()
        face_rec.place(x=20, y=30, width=100, height=20)
        
        self.rootface_emo_signin = tk.IntVar()
        self.rootface_emo_signin.set(0)       # 預設值0=未報到
        face_emo = tk.Checkbutton(self.root, text='心情辨識', variable=self.rootface_emo_signin,onvalue=1, offvalue=0)
        face_emo.place(x=120, y=30, width=100, height=20)
        
        self.rootface_say_signin = tk.IntVar()
        self.rootface_say_signin.set(0)       # 預設值0=未報到
        face_say = tk.Checkbutton(self.root, text='心情小語', variable=self.rootface_say_signin,onvalue=1, offvalue=0)
        face_say.place(x=220, y=30, width=100, height=20)
        
           
        self.lstStudent = tk.Listbox(self.root, width=90)
        self.lstStudent.pack(padx=10, pady=10,side='bottom')
           
        self.panel = tk.Label(self.root)  # initialize image panel
        self.panel.pack(padx=10, pady=10,side='left')   
           
        self.rec_name_cnt=[]
        self.rec_emotion_cnt=[]
          
        
           
        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.video_loop()        
    
        
    def take_makeup(self):            
        os.system('python main_recognition.py')
    
    def video_loop(self):
       # global rec_name_cnt
        self.emotion_labels = {
                0: 'angry',
                1: 'disgust',
                2: 'fear',
                3: 'happy',
                4: 'sad',
                5: 'surprise',
                6: 'neutral'
            }

        def cv2ImgAddText(frame, text, left, top, textColor, textSize):
            if (isinstance(frame, numpy.ndarray)):
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame)
            fontText = ImageFont.truetype(
                    "font/simsun.ttc", textSize, encoding="utf-8")
            draw.text((left, top), text, textColor, font=fontText)
            return cv2.cvtColor(numpy.asarray(frame), cv2.COLOR_RGB2BGR)
        def detect_and_predict_mask(frame, faceNet, maskNet):
            # grab the dimensions of the frame and then construct a blob
            # from it
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
            # pass the blob through the network and obtain the face detections
            faceNet.setInput(blob)
            detections = faceNet.forward()
                    	# initialize our list of faces, their corresponding locations,
            # and the list of predictions from our face mask network
            faces = []
            locs = []
            preds = []
            
            # loop over the detections
            for i in range(0, detections.shape[2]):
                
                # extract the confidence (i.e., probability) associated with
                # the detection
                confidence = detections[0, 0, i, 2]
                    		# filter out weak detections by ensuring the confidence is
                	   # greater than the minimum confidence
#                if confidence > args["confidence"]:
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object                     
                    box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # ensure the bounding boxes fall within the dimensions of
                    # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                    
                    # extract the face ROI, convert it from BGR to RGB channel
                    # ordering, resize it to 224x224, and preprocess it
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    
                    # add the face and bounding boxes to their respective
                    # lists
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))
            
            	 # only make a predictions if at least one face was detected
            if len(faces) > 0:            
            		# for faster inference we'll make batch predictions on *all*
            		# faces at the same time rather than one-by-one predictions
            		# in the above `for` loop
            		faces = np.array(faces, dtype="float32")
            		preds = maskNet.predict(faces, batch_size=32)
            
            	# return a 2-tuple of the face locations and their corresponding
            	# locations
            return (locs, preds)
    
        """ Get frame from the video stream and show it in Tkinter """
#        while(self.vs.isOpened()):\
        # frame就是每一帧的图像，是个三维矩阵
        ok, frame = self.vs.read()  # read frame from video stream
#        # 获取视频的一些参数, 这里是帧速率
#        fps = self.vs.get(5)
#        # 获取视频的总帧数
#        x = self.vs.get(7)
#        print(fps)
#        print(x)
        if ok:  # frame captured without any errors                      
                        
            self.step_while=self.step_while+1
            if self.step_while>30:
                self.x1=0
                self.last_rec_name = ""
                self.step_while = 0  
                self.rec_name_cnt=[]
                self.rec_emotion_cnt=[]
              
            if self.step_while % 1==0: 
                #print(self.step_while)#*************
                
                # OpenCV人脸识别分类器---------------
#                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换灰色           
#                # 调用识别人脸
#                face_rects = classifier.detectMultiScale(
#                    gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))                
#                if len(face_rects):  # 大于0则检测到人脸
#                    for faceRect in face_rects:  
#                        self.x1, self.y1, self.w, self.h = faceRect
#                        self.x2=self.w+self.x1
#                        self.y2=self.h+self.y1
#                        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), ( 0, 255, 0), 2, cv2. LINE_AA)
                #----------------------------------------
                #dlib-----------------------------------   
#                face_rects, self.scores, idx = detector.run(frame, 0)
#                for i, d in enumerate(face_rects):
#                    self.x1 = d.left()
#                    self.y1 = d.top()
#                    self.x2 = d.right()
#                    self.y2 = d.bottom()
##                    print(self.step_while)
##                    print(self.y1)
##                    print(self.y2)
#                    cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), ( 0, 255, 0), 2, cv2. LINE_AA)
                #----------------------------------------
                #face_recognition-----------------------------------
#                face_rects = face_recognition.face_locations(frame)
#                for faceRect in face_rects:  
#                    self.x1, self.y1, self.x2, self.y2 = faceRect
#                   
#                    cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), ( 0, 255, 0), 2, cv2. LINE_AA)
                #faceNet----------------------------------------
                print(faceNet)
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            	# loop over the detected face locations and their corresponding
            	# locations
                for (box, pred) in zip(locs, preds):
                    
                    # unpack the bounding box and predictions
                    (self.x1, self.y1, self.x2, self.y2) = box
                    
                    (mask, withoutMask) = pred
                
                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    
                    # include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                    
                    # display the label and bounding box rectangle on the output
                    # frame
                    #cv2.putText(frame, label, (self.x1, self.y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), color, 2)
                    
                    #box-
                    left = box[0]
                    top = box[1]
                    right = box[2]
                    bottom = box[3] 
#                    print("bottom type", type(box[0]),box[0])
                    
                    d = dlib.rectangle(int(left), int(top), int(right), int(bottom)) 
#                    print(box,type(box))
                    #rec---------------------------------------- 
#                    if self.rootface_rec_signin.get() == True
                    landmarks_frame = cv2.cvtColor(frame, cv2. COLOR_BGR2RGB)
                    shape = predictor(landmarks_frame, d)                    
                    dist = []                    
                    face_descriptor = facerec.compute_face_descriptor(frame, shape)
                    d_test = numpy.array(face_descriptor)
                    for j in descriptors:
                        dist_ = numpy.linalg.norm(j - d_test)
                        dist.append(dist_)
                    c_d = dict( zip(candidate,dist))
                    self.cd_sorted = sorted(c_d.items(), key=lambda kv: kv[1])
                    if self.cd_sorted[0][1]<0.9:#0.4
                        self.rec_name_dist = self.cd_sorted[0][1]
                        self.rec_name = self.cd_sorted[0][0]
                        m = my.explode("#",self.rec_name)
                        self.last_rec_name = m[0]
                        self.last_rec_name = self.last_rec_name.capitalize()
                        if self.last_rec_name!="":
                            frame = cv2ImgAddText(frame,  self.rec_name, self.x1-30, self.y1-25, ( 0, 255, 0), 15)
                            self.result ='第'+str(self.step_while)+'frame,'+self.last_rec_name+ '相似最短距離:'+str(self.rec_name_dist)
                            self.lstStudent.insert(0,self.result)
                    #---------------------------------------- 
            
#                        self.text = " %2.2f ( %d )" % (self.scores[i], idx[i])#******
                        
#                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                        gray_face = gray[(self.y1):(self.y1 + self.y2), (self.x1):(self.x1 + self.x2)]
#                          
##                        print(self.step_while)
##                        print(self.y1)
##                        print(self.y2)                                            
##                        print(gray_face.shape)  
#                        try:
#                            gray_face = cv2.resize(gray_face, (48, 48))
#                            gray_face = gray_face / 255.0
#                            gray_face = numpy.expand_dims(gray_face, 0)
#                            gray_face = numpy.expand_dims(gray_face, -1)
#                            
#    #                            self.emotion_scorce=emotion_classifier.predict(gray_face)
#    #                            print(emotion_scorce)
#                            self.emotion_label_score = numpy.max(emotion_classifier.predict(gray_face))
#                            self.emotion_label_arg = numpy.argmax(emotion_classifier.predict(gray_face))
#    #                        self.emotion = self.emotion_labels[emotion_label_arg]
#    #                            frame = cv2ImgAddText(frame,  self.last_rec_name+'_'+self.text+'_'+self.emotion, self.x1-30, self.y1-25, ( 0, 255, 0), 15)
#                            if self.emotion_label_score>0.01:  #0.5  
#    #                            frame = cv2ImgAddText(frame,  self.rec_name+'_'+self.text+'_'+self.emotion, self.x1-30, self.y1-25, ( 0, 255, 0), 15)
#    #                            self.result ='第'+str(self.step_while)+'frame,'+self.last_rec_name+ '覺得'+ self.emotion+'人臉分數:'+str(self.text)+ '相似最短距離:'+str(round(self.rec_name_dist,2))+'心情分數:'+str(round(self.emotion_label_score,2))
#                               
#                                #self.rec_name_cnt=self.rec_name
#                                #self.rec_name_cnt.append(self.rec_name)
#                                #print(self.rec_name_cnt)
#                                
#                                self.rec_name_cnt.append(self.rec_name)
#                                self.rec_emotion_cnt.append(self.emotion_label_arg)
#                                
#    #                            print(self.rec_name_cnt)
#                                if self.step_while % 30==0:
#                                    
#                                    self.rec_name_max=max(self.rec_name_cnt,key=self.rec_name_cnt.count)                                
#                                    self.arr_appear=dict((a,self.rec_name_cnt.count(a)) for a in self.rec_name_cnt)
#                                    print(self.arr_appear)
#                                    print(self.rec_name_max)
#                                    
#                                    self.rec_emotion_max=max(self.rec_emotion_cnt,key=self.rec_emotion_cnt.count)
#                                    self.arr_appear_emo=dict((a,self.rec_emotion_cnt.count(a)) for a in self.rec_emotion_cnt)
#                                    print(self.arr_appear_emo)
#                                    print(self.rec_emotion_max)
#                                    
#                                    self.emotion = self.emotion_labels[self.rec_emotion_max]
#                                    
#    #                                self.arr_appear_sorted = sorted(self.arr_appear.items(), key=lambda kv: kv[1])
#    
#                                
#    #                                self.result =str(self.step_while)+self.last_rec_name+'_'+ self.emotion+'人臉分數:'+str(self.text)+ '相似距離:'+str(round(self.rec_name_dist,2))+'心情分數:'+str(round(self.emotion_label_score,2))
#                                    self.result =str(self.step_while)+self.rec_name_max+'_'+ self.emotion+'人臉分數:'+str(self.text)+ '相似距離:'+str(round(self.rec_name_dist,2))+'心情分數:'+str(round(self.emotion_label_score,2))
#                                    
#                                    self.lstStudent.insert(0,self.result)
#                                    frame = cv2ImgAddText(frame, self.result, 5, self.y1-25, ( 255, 0, 0),20)
#                                    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # grab the current timestamp   
#                                    filename ="{}.jpg".format(ts)  # construct filename
#                                    p = os.path.join(np_dir,self.last_rec_name+"_"+ filename)  # construct output path  self.output_path     
#                                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
#                                    self.current_image = Image.fromarray(cv2image) 
#                                    self.current_image.convert("RGB").save(p,"JPEG")  # save image as jpeg file            
#                                    fp.writelines(self.result+"\n")
#                                    fp.flush()
#        
#        
#    #                                if emotion_label_arg ==0 :
#                                    if self.rec_emotion_max ==0 :
#    #                                    engine.say(self.last_rec_name + '早安，祝您有美好的一天')
#                                        engine.say(self.rec_name_max + '早安，祝您有美好的一天')
#                                    elif self.rec_emotion_max ==1 :
#                                        engine.say(self.rec_name_max + '嗨!記得吃早餐喔')
#                                    elif self.rec_emotion_max ==2 :
#                                        engine.say(self.rec_name_max + '今天替你加加油')
#                                    elif self.rec_emotion_max ==3 :
#                                        engine.say(self.rec_name_max + '要一值保持開心喔')
#                                    elif self.rec_emotion_max ==4 :
#                                        engine.say(self.rec_name_max + '早安，期待美好的一天')                        
#                                    elif self.rec_emotion_max ==5 :
#                                        engine.say(self.rec_name_max + '')
#                                    elif self.rec_emotion_max ==6 :
#                                        engine.say(self.rec_name_max + ' 早!買杯咖啡喝吧')
#        #                            else:
#        #                                engine.say(self.last_rec_name + '您好，您尚未成為我們的VIP，請快加入我們吧') 
#                                    engine.runAndWait()  
#
#   
#                        except:
#                            continue 
            
                #show text-----------------            
#                #if self.x1!=0:                      
#                   # cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), ( 0, 255, 0), 2, cv2. LINE_AA)

#                if self.last_rec_name!="":
##                    frame = cv2ImgAddText(frame,  self.rec_name+'_'+self.text+'_'+self.emotion, self.x1-30, self.y1-25, ( 0, 255, 0), 15)
##                    self.result ='第'+str(self.step_while)+'frame,'+self.last_rec_name+ '覺得'+ self.emotion+'人臉分數:'+str(self.text)+ '相似最短距離:'+str(self.rec_name_dist)+'心情分數:'+str(self.emotion_label_score)
#                    
#                    frame = cv2ImgAddText(frame,  self.rec_name, self.x1-30, self.y1-25, ( 0, 255, 0), 15)
#                    self.result ='第'+str(self.step_while)+'frame,'+self.last_rec_name+ '相似最短距離:'+str(self.rec_name_dist)
#                   
#                    self.lstStudent.insert(0,self.result)
                #------------------------------
                    #save------------------
#                    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # grab the current timestamp   
#                    filename ="{}.jpg".format(ts)  # construct filename
#                    p = os.path.join(np_dir,self.last_rec_name+"_"+ filename)  # construct output path  self.output_path     
#
#                    self.current_image.convert("RGB").save(p,"JPEG")  # save image as jpeg file    
#                    fp.writelines(self.result+"\n")
#                    fp.flush()
                    #------------------
  
            
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
#            self.mmm=self.current_image           
            
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            self.panel.config(image=imgtk)  # show the image
            self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds           
    
  
      

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="./",
       help="path to output directory to store snapshots (default: current folder")
args = vars(ap.parse_args())

# start the app
print("[INFO] starting...")
pba = Application(args["output"])
pba.root.mainloop()
fp.close()