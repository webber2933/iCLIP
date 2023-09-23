## Data Preparation

### Easy Version for J-HMDB

1. Download the zip file from [[here]](https://drive.google.com/file/d/1rwj3qo4r25e1_a3zB8kKwjKh1kFQCmU8/view?usp=sharing). 

2. run following commands to unzip the file and create a 
symbolic link to the extracted files.

    ```bash
    cd /path/to/iCLIP
    mkdir data
    unzip jhmdb.zip -d iCLIP/data/
    ```

3. To do the zero-shot train/inference, put the train videos/labels and test videos/labels in ```data/jhmdb/label_split```. You can refer to the following files, which are used for zero-shot training/inference in 75%vs25% labels split:
**[train_label.txt](https://drive.google.com/file/d/1Rdu9e2Efi9Lp2_eSVI-v45mn6pgmpiKs/view?usp=sharing)**
**[train_video.txt](https://drive.google.com/file/d/1k85KqkxqwfAXoeDgXpA1FiOIWm_jAZCP/view?usp=sharing)**
**[test_label.txt](https://drive.google.com/file/d/1kRTeCgsaA0UgALuu_7QeTbgejIA8TGWp/view?usp=sharing)**
**[test_video.txt](https://drive.google.com/file/d/1e9Z-9ElSyL8QnK7NAQksQk0z7vjATHx5/view?usp=sharing)**
