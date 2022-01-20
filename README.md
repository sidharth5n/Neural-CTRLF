# Neural-CTRLF
PyTorch implementation of [Neural Ctrl-F: Segmentation-free Query-by-String Word Spotting in Handwritten Manuscript Collections](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wilkinson_Neural_Ctrl-F_Segmentation-Free_ICCV_2017_paper.pdf), ICCV, 2017.

Key differences from the [original implementation](https://github.com/tomfalainen/neural-ctrlf) :
1. The CNN backbone used is ResNet34 instead of Pre Activated ResNet34.
2. Input images are in RGB format instead of gray scale.
3. 128 filters in RPN head instead of 256.

## Dependencies


## Data Preparation
Download Washington dataset from [here](http://ciir.cs.umass.edu/downloads/gw/gw_20p_wannot.tgz) and extract the contents to `data/washington` or run the following code.
```
mkdir -p data/washington/
cd data/washington
wget http://ciir.cs.umass.edu/downloads/gw/gw_20p_wannot.tgz
tar -xzf gw_20p_wannot.tgz
cd ../../
```
Prepare the dataset by running the following code. `augment` and `cross_val` can be set for data augmentation and 4-fold cross validation respectively.
```
python preprocess.py --augment False --cross_val False --embedding dct
```
## Training
First, download model checkpoint pre-trained on [IIIT-HWS dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/matchdocimgs) from [here]() and place it in `$root/checkpoints` directory. Now, run the following code with the same settings used for preparing the dataset.
```
python train.py --id resnet34 --augment False --cross_val False --embedding dct
```
The model checkpoints, loss dumps and infos will be saved at `checkpoints/$id/`. For a list of all the hyper parameters used for training, refer [opts.py]().

## Testing
```
python test.py --id resnet34 --split test
python evaluate_map.py
```

## Evaluation
For querying the word 'hello' on an image located at `$image_path`, run the following code
```
python eval.py --id resnet34 --query hello --image $image_path
```

## Model Zoo

## Citation
If you find this repository useful, please consider citing the Neural Ctrl-F paper.
```
@INPROCEEDINGS{Wilkinson2017,
  author = {Wilkinson, Tomas and Lindström, Jonas and Brun, Anders},
  booktitle = {2017 IEEE International Conference on Computer Vision (ICCV)}, 
  title = {Neural Ctrl-F: Segmentation-Free Query-by-String Word Spotting in Handwritten Manuscript Collections}, 
  year = {2017},
  pages = {4443-4452},
}
```