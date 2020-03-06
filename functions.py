import cv2
import numpy as np 
import pickle
import imutils
import os
from PIL import Image
import sys

def main(model):
    print("Options :-")
    print("\t1) Recognize a Face ")
    print("\t2) Add (or update) a new face using webcam ")
    print("\t3) Add (or update)image from path ")
    print("\t4) Exit ")

    opt = int(input("\nSelect an option : "))
    
    exec_opt(opt,model)

#convert an image file to its corresponding 128 dimentional vector face encoding
def img_to_enc(img,model):
    image = img.resize((96, 96))
    face_array = np.asarray(image).reshape((1,96,96,3))
    face_array = np.transpose(face_array,axes=[0,3,1,2])
    return model.predict(face_array)

#required face detection model
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)

# add or update a new face to the database
def add_face_to_dict(name,image,model):
    face_dict=load_dict()
    if name in face_dict.keys():
        print(" Face Already in Database, Updating ...")
    face_dict[name]=img_to_enc(image,model)
    pickle_face=open("face_dict.pickle","w+b")
    pickle.dump(face_dict,pickle_face)
    pickle_face.close()

# function to capture a face using webcam
def add_face_from_cam(model):
    name=input("Enter the name of the person : ")
    print("Add a new face")
    camera=cv2.VideoCapture(0+cv2.CAP_DSHOW)
    ret_st=1
    i=0
    while True:
        suc,frame=camera.read()
        if suc==False:
            i+=1
            if i==100:
                camera.release()
                cv2.destroyAllWindows()
        else:
            frame=imutils.resize(frame,width=500)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            if len(faces)>0:
                (x,y,w,h) = faces[0]
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.imshow("Press Q to select face (c to cancel )",frame)
            key=cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                add_face_to_dict(name,Image.fromarray(frame[y:y+h,x:x+w]),model)                
                break
            elif key & 0xFF ==ord('c'):
                ret_st=0
                break
    camera.release()
    cv2.destroyAllWindows()
    if ret_st==0 :
        print("Face not added ...")

# add an image from a path
def add_image_from_path(model):
    path=input("Enter the path of the image : ")
    name=input("Enter the name : ")
    img=Image.open(path)
    add_face_to_dict(name,img,model) 

# to load the face database
def load_dict(path="face_dict.pickle",mode="rb"):
    if os.path.getsize(path) == 0:
        print("Empty Database")
        return {}
    pick_=open(path,mode)
    dic=pickle.load(pick_)
    pick_.close()
    return dic

#to compare the similarity between two faces 
def check(enc,face_dict,threshold):
    min_dist=1
    best_face = None
    for name in face_dict.keys():
        orm = np.linalg.norm(face_dict[name]-enc,ord=2)
        if orm<threshold:
            if orm<min_dist:
                min_dist=orm
                best_face=name
    return best_face,min_dist

# function to open the webcam and recognize the person in front of it 
def recog(model,threshold=0.07):
    
    camera=cv2.VideoCapture(0+cv2.CAP_DSHOW)
    prev=None
    face_dict=load_dict()
    print("Recognizing ...")
    while True:
        ret,frame=camera.read()
        if prev != ret:
            print(ret)
        prev=ret
        if ret:    
            frame=imutils.resize(frame,width=500)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            if len(faces)>0:
                for (x,y,w,h) in faces:
                    roi_color=frame[y:y+h,x:x+w]
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    enc=img_to_enc(Image.fromarray(roi_color),model)
                    best_match,orm=check(enc,face_dict,threshold)
                    if best_match==None:
                        cv2.putText(frame,"No Match Found",(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.45,(255,0,0),2)
                    else:
                        cv2.putText(frame,best_match,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.45,(255,0,0),2)
                    print(best_match,orm)
            
            cv2.imshow("Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            continue
    camera.release()
    cv2.destroyAllWindows()

def exec_opt(opt,model):
    if opt==1 :
        recog(model)        
    elif opt==2 :
        add_face_from_cam(model)        
    elif opt==3:
        add_image_from_path(model)
    elif opt==4 :
        sys.exit('Ending ...')
    else :
        print(" Enter a valid option (1 to 4)")
    return

