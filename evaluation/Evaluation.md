# Evaluation

We use the following commands to test the frame mAP metric. 

## 1. Obtain csv result file

First we need to obtain the result of detection (i.e., the output of inference). In our code, we will put it in ```data/output/dense_serial_debug/inference/jhmdb_val/result_jhmdb.csv``` after inference, or you can also provide a detection result yourself.

The detection result should be a csv file, in which the format of each row is as follows:
```bash
video_name, timestamp, bbox(x1,y1,x2,y2), label_index, score
ex: pullup/30_(dead-hang)_pull-ups_pullup_u_cm_np1_ba_med_1,1,123,19,240,239,1,0.5
```

## 2. Make pkl file of detections
You can use the following command to convert csv result into pkl file:
```bash
python make_frame_detection_zero_shot.py
```
Note that you should provide the (1) test_label (2) groundtruth (3) csv result in ```make_frame_detection_zero_shot.py```. The code will output a pkl file for evaluation.

## 3. Evaluation
You can use the following command to evaluate the frame mAP on test labels:
```bash
python evaluate_zero_shot.py frameAP [groundtruth] [detection pkl file]
ex: python evaluate_zero_shot.py frameAP JHMDB-GT.pkl frame_detections.pkl
```
Note that you should provide the train_label in the ```frameAP``` function of ```evaluate_zero_shot.py```, which will prevent the code from evaluating the AP of these training classes.

## Acknowledgement
The ```evaluate_zero_shot.py``` is modified from [MultiSports](https://github.com/MCG-NJU/MultiSports). We thank the authors for open-sourcing their code.