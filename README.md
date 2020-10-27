# out-of-distribution-detection

This repo is a pytorch implementation of [Out-of-Distribution Detection Using an Ensembleof Self Supervised Leave-out Classifiers](https://arxiv.org/pdf/1809.03576.pdf) 
realised within the course [DD2412 Deep Learning, Advanced](https://www.kth.se/student/kurser/kurs/DD2412?l=en) at KTH university.

## Data

* [CIFAR10](https://course.fast.ai/datasets)
* [Tiny-ImageNet (crop)](https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz) (provided by [fb-research](https://github.com/facebookresearch/odin))
* [Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz) (provided by [fb-research](https://github.com/facebookresearch/odin))
* [LSUN (crop)](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz) (provided by [fb-research](https://github.com/facebookresearch/odin))
* [LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz) (provided by [fb-research](https://github.com/facebookresearch/odin))
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz) (provided by [fb-research](https://github.com/facebookresearch/odin))

Uniform Noise (UNFM) and Gaussian Noise (GSSN) can be generated thanks to the script ```noise_ood_datasets_generator.py```

## Project structure

The project is structured as following:

```code
.
├── data # data should be upload here
├── datasets
|   └──  cifar10.py # a custom dataset design to handle margin_loss with cifar10
|   └──  ood.py # basic custom dataset for ood data
├── distributions # has to be created, distribution plots (id vs ood) will be saved here
├── models
|   └──  dense_net.py # densenet nn
|   └──  toy_net.py # a basic cnn taken from the pytorch cifar10 tutorial
|   └──  wide_res_net.py # wideresnet nn
├── distributions # has to be created, models weights will be saved here
├── class_to_id_lists.py
├── classifier.py # definition of the Classifier class
├── loss.py # entropy based margin loss
├── metrics.py # detection error, fpr95, auroc, aupr in, aupr out
├── noise_ood_datasets_generator.py # has to be run to generate uniform noise and gaussian noise datasets
├── ood_validation.py # ood scores computation
├── train.py # pipelines for training
├── test.y # pipelines for testing
```

## Launching

Train: execute ```train.py```, parameters have to be chosen into the file.  
Validation is executed on iSUN with T=100, epsilon=0.002.

Test: execute ```test.py```, net_architecture and training_name have to be consistent.  
The results have been obtained by executing this file as is.

Project made by Simon Lembeye, Emile Lucas and Muhammed Memedi.