## Getting Started with iCLIP

We recommend to create a directory `models` inside `iCLIP/data` to place 
model weights. 

```shell
cd /path/to/iCLIP
mkdir data/models
ln -s /path/to/models data/models
```

### Training

Download pre-trained models from [MODEL_ZOO.md](MODEL_ZOO.md#pre-trained-models).
Then place pre-trained models in `data/models` directory with following structure:

```
models/
|_ pretrained_models/
|  |_ SlowFast-ResNet50-4x16.pth
```

Train on a single GPU:

```shell
python train_net.py --config-file "config_files/iCLIP.yaml"
```


### Inference

After training, the model will be placed in `OUTPUT_DIR`. Note that our code tries to load the `last_checkpoint` in the `OUTPUT_DIR`. 
You can also download the pretrained weights from [MODEL_ZOO.md](MODEL_ZOO.md#ava-models), then put it in `OUTPUT_DIR`.
 
Run the following command to perform inference.
 ```shell
python test_net.py --config-file "config_files/iCLIP.yaml"
 ```
The output files will be written to `OUTPUT_DIR/inference/jhmdb_val/detections` and `OUTPUT_DIR/inference/jhmdb_val/result_jhmdb.csv`.