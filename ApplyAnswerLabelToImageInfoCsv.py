# 顔認識結果が記録されたcsvファイルに正解ラベル情報を適用する。

import cv2
import sys, glob, os
from PIL import Image
import matplotlib.pyplot as plt

def main():
  if len (sys.argv) == 5:
    input_csv_path = sys.argv[1]
    input_resized_images_dir = sys.argv[2]
    input_labeled_images_dir = sys.argv[3]
    output_images_path = sys.argv[4]
  else:
    print('Usage: input.csv labeled_images_dir output.csv')
    sys.exit(1)
  #

  # あらかじめinput_labeled_images_dirを走査し、ラベルを一覧にしておく
  files = os.listdir(input_labeled_images_dir)
  labels = [f for f in files if os.path.isdir(os.path.join(input_labeled_images_dir, f))]
  print(labels)

  count_map = {}
  for label in labels:
    if not os.path.exists(output_images_path + '\\' + label):
      os.mkdir(output_images_path + '\\' + label)
    #
    count_map[label] = 0
  #

  with open(input_csv_path) as inputf:
    cnt = 0
    for line in inputf:
      line_strip = line.strip()
      line_list = line_strip.split(',')

      for label in labels:
        if os.path.exists(input_labeled_images_dir + '\\' + label + '\\' + line_list[0]):
          print ('[' + str(cnt) + '] ' + label)
          count_map[label] += 1

          original_filepath = input_resized_images_dir + '\\' + (line_list[0])[9:]
          image_org = cv2.imread(original_filepath)
          print(original_filepath)

          # 画像を切り抜いて保存する         
          x_start = int(line_list[1])
          x_end = x_start + int(line_list[3])
          y_start = int(line_list[2])
          y_end = y_start + int(line_list[4])
          print(str(x_start) + ',' + str(x_end) + ',' + str(y_start) + ',' + str(y_end))

          image_cropped = image_org[y_start:y_end, x_start:x_end]
          out_img_filepath = output_images_path + '\\' + label + '\\' + line_list[0]
          print(out_img_filepath)

          cv2.imwrite(out_img_filepath, image_cropped)
        #
      #
      cnt += 1
    #
  #

  print('----------------')
  for label in labels:
    print(label + ' = ' + str(count_map[label]))
  #

  return 0

#

if __name__ == "__main__":
  main()
#