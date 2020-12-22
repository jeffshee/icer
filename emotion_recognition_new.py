import pandas as pd
import cv2
import numpy as np
from os.path import join
from keras.models import model_from_json
from ast import literal_eval as make_tuple
import face_recognition
from scipy.spatial import distance as dist


# MAR: Mouse Aspect Ratio
# https://medium.freecodecamp.org/smilfie-auto-capture-selfies-by-detecting-a-smile-using-opencv-and-python-8c5cfb6ec197
def calculate_MAR(landmarks):
    A = dist.euclidean(landmarks["top_lip"][2], landmarks["bottom_lip"][4])
    B = dist.euclidean(landmarks["top_lip"][3], landmarks["bottom_lip"][3])
    C = dist.euclidean(landmarks["top_lip"][4], landmarks["bottom_lip"][2])
    L = (A + B + C) / 3  # ABCの平均→縦
    D = dist.euclidean(landmarks["top_lip"][0], landmarks["bottom_lip"][0])  # 横
    mar = L / D
    return mar

def get_parts_coordinates(a, b):
    l = list()
    if b == True:
        for w in a:
            l.append(w[0])
    elif b == False:
        for w in a:
            l.append(w[1])
    return l

def get_video_dimension(video_capture):
    return int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

def get_face_location_from_csv(df, frame_num):
    face_location = df.loc[df["frame"] == frame_num].values.tolist()
    if len(face_location) == 0:
        # frame_num not found in csv
        return None
    return [make_tuple(s) for s in face_location[0][2:]]

def get_eye_location(face_landmarks):
    # 目の位置
    leye_x = get_parts_coordinates(face_landmarks["left_eye"], True)
    reye_x = get_parts_coordinates(face_landmarks["right_eye"], True)
    eye_y = get_parts_coordinates(face_landmarks["left_eye"] + face_landmarks["right_eye"], False)
    topeye = min(eye_y)
    bottomeye = max(eye_y)
    lefteye = min(leye_x)
    righteye = max(reye_x)
    eye_center=  (righteye + lefteye) // 2, (topeye + bottomeye) // 2
    return eye_center

def emotion_recognition(video_path,batch_size,k_resolution,file_path):
    df = pd.read_csv(file_path) ##读取csv文件
    # print(df["face1"][3])
    k=3 ##每3帧检测一次
    model_dir = '../model'
    model_name = 'mini_XCEPTION'
    model = model_from_json(open(join(model_dir, 'model_{}.json'.format(model_name)), 'r').read())
    model.load_weights(join(model_dir, 'model_{}.h5'.format(model_name)))##载入模型

    # Open video file
    video_capture = cv2.VideoCapture(video_path) ##读取视频
    original_w, original_h = get_video_dimension(video_capture) ##获取视频宽度和长度
    resize_rate = (1080 * k_resolution) / original_w  # Resize
    w = int(original_w * resize_rate)
    h = int(original_h * resize_rate)

    frames = []
    faces = []
    time_frame=[]
    count=0
    last_count=0
    frame_count = 0
    start_frame=0
    emotion = {}
    eye_center={}
    mouse_opening_rate_list={}
    person_number=3 ## 参加的人数
    for i in range(0,person_number):
        emotion[i]={}
        eye_center[i]={}
        mouse_opening_rate_list[i]={}
    while video_capture.isOpened():
        # Grab a single frame of video
        ret, frame = video_capture.read() ##从视频流中读取一帧

        # Bail out when the video file ends
        if not ret:
            break

        face_location = get_face_location_from_csv(df, frame_count)  ##获取某一帧的所有信息 形成一个list
        # print(face_location)

        for top, right, bottom, left in face_location:
            face = frame[top:bottom, left:right]
            import matplotlib.pyplot as plt
            plt.imshow(face)
            plt.show()
            face = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (64, 64)) / 255.0
            faces.append(face)
        frame_count += 3 ##3帧进行一次表情识别
        count +=1 ##进行一次计数
        # Save each frame of the video to a list
        frames.append(frame)

        # #save the time
        # time_frame.append(frame_count) ##计算时间

        # Every 128 frames (the default batch size), batch process the list of frames to find faces
        if count%128==0:
            val = np.stack(faces)
            predictions = model.predict(val)
            # print(predictions)
            # Clear the frames array to start the next batch
            emotion_count=0##从0开始计数
            for i in range(last_count,count): ##帧数
                time_frame.append(start_frame + i * k)##计算当前帧的时间
                for j in range(0,person_number): ##人数
                    j_location=get_face_location_from_csv(df,start_frame+i*k)[j]
                    if j_location!="Unknow":
                        emotion[j][start_frame + i * k]=predictions[emotion_count]

                        ##判断眼睛坐标
                        face_landmarks = face_recognition.face_landmarks(face[emotion_count], face_locations=[
                            (0, face[emotion_count].shape[1], face[emotion_count].shape[0], 0)])
                        tmp_eye_center=get_eye_location(face_landmarks)
                        eye_center[j][start_frame + i * k]=j_location[3]+tmp_eye_center[0],j_location[0]+tmp_eye_center[1]
                        mouse_opening_rate_list[j][start_frame + i * k] =calculate_MAR(face_landmarks)
                        emotion_count=emotion_count+1
                    else:
                        emotion[j][start_frame + i * k]="Unknow"
                        eye_center[j][start_frame + i * k]=0,0  ##无法识别表情的时候不写眼的坐标
                        mouse_opening_rate_list[j][start_frame + i * k]=0
            last_count=count
            start_frame=start_frame+(count-last_count)*k
            faces = []

# a=emotion_recognition("file/out1.mp4",3,128,"file/detect_face.csv")