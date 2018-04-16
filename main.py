import os
from joblib import Parallel, delayed
import cv2
import dlib

file_dir = "./raw/"
icon_dir = "./icon/"
# 顔認識特徴量ファイル
face_detector = dlib.simple_object_detector("./detector_face.svm")
# 目認識特徴量ファイル
eye_detector = dlib.simple_object_detector("./detector_eye.svm")

# 指定されたフォルダー内にあるフォルダーの一覧生成
def listdir(folder):
    return_list = []
    for dir_list in os.listdir(folder):
        if os.path.isdir(folder+dir_list):
            return_list.append(folder+dir_list)
    return return_list

def icon_maker(target):
    # 現在のアイコンサイズ
    max_size = 0
    # アイコン名
    icon_name = target.split("/")[-1]
    for temp in os.listdir(target):
        filename = target + "/" + temp
        _, ext = os.path.splitext(temp)
        image = cv2.imread(filename)
        height, width, channels = image.shape
        faces = face_detector(image)
        # 顔が検出できたか
        if len(faces) > 0:
            for i, rect in enumerate(faces):
                # エリア拡大
                size = rect.bottom() - rect.top()
                ys = int(rect.top() - size/1.5)
                if(ys < 0):
                    xs = 0
                ye = int(rect.bottom() + size/1.5)
                if(ye > height):
                    xe = height
                size = rect.right() - rect.left()
                xs = int(rect.left() - size/1.5)
                if(xs < 0):
                    xs = 0
                xe = int(rect.right() + size/1.5)
                if(xe > width):
                    xe = width
                # 現状のアイコンより大きいか
                if max_size < size:
                    # 顔だけ切り出し
                    dst = image[ys:ye, xs:xe]
                    eyes = eye_detector(dst)
                    # 目が検出できたか
                    if len(eyes) > 0:
                        new_image_path = icon_dir + icon_name + ext
                        cv2.imwrite(new_image_path, dst)
                        max_size = size
    print("maked : " + icon_name)

def main():
    Parallel(n_jobs=-1)([delayed(icon_maker)(folder) for folder in listdir(file_dir)])

if __name__ == "__main__":
    main()