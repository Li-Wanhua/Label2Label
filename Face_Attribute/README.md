# Face Attribute Recognition

## Get Started

- Data prepare
  [LFWA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  [train_lfwa.txt](https://pan.baidu.com/s/1ze2Vi48VY-fmSWiPKlBsaQ?pwd=hegy)
  [val_lfwa.txt](https://pan.baidu.com/s/14Dr_E1BjuEzTq-8DwwFQtg?pwd=33w8)
  
  Prepare datasets to have following structure:
  
  ```shell
    Face_Attribute/dataset
        LFWA
            lfw/
            train_lfwa.txt
            val_lfwa.txt
  ```

- Training scripts

  ``` shell
  # resnet50
  python main.py --randAug --bn1d --back resnet50 --arch L2L --dataset LFWA --head norm -p 391
  # you should get error=12.51
  # resnet101
  python main.py --randAug --bn1d --back resnet101 --arch L2L --dataset LFWA --head norm -p 391
  # you should get error=12.44
  ```

- Evaluate scripts

  ``` shell
  # resnet50
  python main.py --randAug --bn1d --back resnet50 --arch L2L --dataset LFWA --head norm -p 391 --eval-only --eval-ckpt $MODEL_PATH
  # you should get error=xx.xx
  # resnet101
  python main.py --randAug --bn1d --back resnet101 --arch L2L --dataset LFWA --head norm -p 391 --eval-only --eval-ckpt $MODEL_PATH
  # you should get error=xx.xx
  ```

  |Backbone|Error|Log|Weight|
  |--|--|--|--|
  |Resnet50|12.51|[log_resnet50](https://cloud.tsinghua.edu.cn/f/124a549712414c51901d/?dl=1)|[model_resnet50](https://cloud.tsinghua.edu.cn/f/1cc36fa2d0c34c95a7d2/?dl=1)|
  |Resnet101|12.44|[log_resnet101](https://cloud.tsinghua.edu.cn/f/a9ec625eef1347c6b003/?dl=1)|[model_resnet101](https://cloud.tsinghua.edu.cn/f/9fd0818f47a549d0861e/?dl=1)|

&emsp;you may notice that we use a specific random seed number, you can also use another seed number by `--seed $NUM`, or you can set seed randomly by `--seed 0`. Our method can get similar performance with different random seed. Here we just fix the seed so that you can get the result exactly match one of our experiment log.
