import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QFileDialog


def get_video_path():
    _ = QApplication(sys.argv)
    video_path = ""
    while not video_path:
        video_path = QFileDialog.getOpenFileName(caption="動画を指定してください", filter="Videos (*.mp4 *.avi)")[0]
    return video_path


def get_face_num():
    _ = QApplication(sys.argv)
    i, ret = None, False
    while not ret:
        i, ret = QInputDialog.getInt(QWidget(), "REID作成", "人数を入力してください", min=1, max=10)
    return i

def get_transcript_index(number):
    _ = QApplication(sys.argv)
    trans_list = []
    i, ret = None, False
    while not ret:
        i, ret = QInputDialog.getText(QWidget(), "index修正", "transcriptの対応indexを入力してください")
    if len(i)==number:
        for j in range(number):
            if str(j) not in i:
                return trans_list
    else:
        return trans_list
    for j in range(len(i)):
        trans_list.append(int(i[j]))
    return trans_list

## test
if __name__ == '__main__':
    while(1):
       tmp_list= get_transcript_index(5)
       # print(tmp_list)
       if len(tmp_list)==0:
           print(len(tmp_list))
           continue
       else:
           break
    print("over")