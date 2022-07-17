# Pedestrian Attribute Recognition

## Introduction

&emsp;Most training code borrow from [Strong_Baseline_of_Pedestrian_Attribute_Recognition](https://github.com/aajinjin/Strong_Baseline_of_Pedestrian_Attribute_Recognition)

## Dataset Info

 PA100K[[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_HydraPlus-Net_Attentive_Deep_ICCV_2017_paper.pdf)][[Github](https://github.com/xh-liu/HydraPlus-Net)]

- Prepare datasets to have following structure:

    ```shell
    Pedestrian_Attribute/data
        PA100k
            data/
            annotation.mat
            README.txt
     ```

- Training on PA100K:

    ```shell
    python train.py
    ```