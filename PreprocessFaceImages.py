# 正解データ作成に先立ち、写真から顔領域を抽出しannotateする。

import cv2
import sys, glob, os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

RECTANGLE_COLOR = (0, 0, 255)
RESIZE_FACTOR = 0.2

def main():
  if len(sys.argv) == 4:
    input_dir_path = sys.argv[1]
    output_annotated_dir_path = sys.argv[2]
    output_resized_dir_path = sys.argv[3]
  else:
    print('Usage: input_dir output_dir output.csv')
    sys.exit(1)
  #

  # 顔認識用特徴ファイルを読み込む
  cascade = cv2.CascadeClassifier(".\\data\\haarcascades\\haarcascade_frontalface_default.xml")

  output_csv_filepath = output_annotated_dir_path + "\\facedetect_results.csv"

  with open(output_csv_filepath, ('a')) as outf:
    cnt = 0
    for f in glob.glob(input_dir_path + '\\*.jpg'):
      print(f)

      # 画像の読み込み
      image_org = cv2.imread(f)
      image = cv2.resize(image_org, (int(image_org.shape[1] * RESIZE_FACTOR), int(image_org.shape[0] * RESIZE_FACTOR)))
      height, width, channels = image.shape[:3]

      # グレースケールに変換
      image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # パラメータの定義：顔領域は画像の縦横小さいほうの1/10よりも大きいものとする
      min_size = int(float(min([width, height])) / 10.0)
      print('min_size = ' + str(min_size))

      # # 入力画像の表示
      # plt.figure(figsize=(8,8), dpi=100)
      # plt.show(plt.imshow(image_gs, cmap="gray"))

      # 顔が認識されたら、認識結果をcsvファイルに書き込むとともに画像として出力する
      face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=1, minSize=(min_size,min_size))

      if len(face_list) == 0:
        continue
      #

      img_filename = os.path.basename(f)

      # todo 何らかの顔が認識された画像（縮小済みのもの）を所定のディレクトリに出力する
      out_resized_filepath = output_resized_dir_path + "\\" + img_filename
      cv2.imwrite(out_resized_filepath, image)

      cnt_local= 0
      for face in face_list:
        x,y,w,h = face

        # csvファイルに書き出し
        out_img_filename = str(cnt).zfill(5) + "_" + str(cnt_local).zfill(2) + "_" + img_filename
        outf.write(out_img_filename + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + ',dummylabel\n')

        # 画像ファイルに書き出し
        image_tmp = image.copy()
        cv2.rectangle(image_tmp, (x,y), (x+w, y+h), RECTANGLE_COLOR, thickness=3)
        out_img_filepath = output_annotated_dir_path + "\\" + out_img_filename
        cv2.imwrite(out_img_filepath, image_tmp)

        cnt += 1
        cnt_local += 1
    #
  #
  return 0
#

if __name__ == "__main__":
  main()
#