# 顔認識結果が記録されたcsvファイルに正解ラベル情報を適用する。

import sys, glob, os

def main():
  if len (sys.argv) == 4:
    input_csv_path = sys.argv[1]
    input_labeled_images_dir = sys.argv[2]
    output_csv_path = sys.argv[3]
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
    count_map[label] = 0
  #

  with open(output_csv_path, ('w')) as outf:
    with open(input_csv_path) as inputf:
      cnt = 0
      for line in inputf:
        line_strip = line.strip()
        line_list = line_strip.split(',')
        line_strip_replaced = line_strip

        for label in labels:
          if os.path.exists(input_labeled_images_dir + '\\' + label + '\\' + line_list[0]):
            print ('[' + str(cnt) + '] ' + label)
            line_strip_replaced = line_strip.replace('dummylabel', label)
            count_map[label] += 1
          #
        #
        outf.write(line_strip_replaced + '\n')
        cnt += 1
      #
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