import pickle
import csv
import numpy as np
from tqdm import tqdm

labels = ['brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'stand', 'swing_baseball', 'throw', 'walk', 'wave']
test_label_file = open("/home/deeperAction22/ActionRec/webber/jhmdb_label_split/75vs25/1/test_label.txt", "r")
data = test_label_file.read()
our_test_labels = data.split("\n")
our_test_labels = our_test_labels[:-1]
test_label_file.close()

objects = []

with (open("JHMDB-GT.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile,encoding='latin1'))
        except EOFError:
            break
print(objects[0].keys())
print(objects[0]['test_videos'][0])
testlist = objects[0]['test_videos'][0]

with open("/home/deeperAction22/ActionRec/webber/current_research/jhmdb/zero_shot/ablation study/LoRA/LoRA_rank_4/data/output/dense_serial_debug/inference/jhmdb_val/result_jhmdb.csv", newline='') as csvfile:
    rows = csv.reader(csvfile)
    
    label_indexes = []
    data = []
    for i in tqdm(rows):
        if float(i[-1]) > 0: # score threshold
            video_index = testlist.index(i[0])
            frame_number = int(i[1])
            x1, y1, x2, y2 = (i[2], i[3], i[4], i[5])
            our_label_index = int(i[6]) - 1
            ori_label_index = labels.index(our_test_labels[our_label_index])
            score = float(i[7])
        
            item = np.array([video_index, frame_number, ori_label_index, score, x1, y1, x2, y2])
            data.append(item)
            label_indexes.append(ori_label_index)

print(set(sorted(label_indexes)))
data = np.array(data, dtype='float32')      
with open('frame_detections.pkl', 'wb') as f:
    pickle.dump(data, f, protocol=2)

print("Done")