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

