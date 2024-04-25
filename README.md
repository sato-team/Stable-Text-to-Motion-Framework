# SATO: Stable Text-to-Motion Framework

[Wenshuo chen](https://github.com/shurdy123)
<!-- **International Conference on 3D Vision (3DV 2024)** -->

[![Website shields.io](https://img.shields.io/website?url=http%3A//poco.is.tue.mpg.de)]() [![YouTube Badge](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)]()  [![arXiv](https://img.shields.io/badge/arXiv-2308.12965-00ff00.svg)]()  


<!-- <div style="display:flex;">
    <img src="assets/run_lola.gif" width="45%" style="margin-right: 1%;">
    <img src="assets/yt_solo.gif" width="45%">
</div> -->


## Setup and Installation

Clone the repository: 
```shell
git clone https://github.com/sato-team/Stable-Text-to-motion-Framework.git
```

Create fresh conda environment and install all the dependencies:
```
conda env create -f environment.yml
conda activate SATO
```
The code was tested on Python 3.8 and PyTorch 1.8.1.
## Dependencies
```shell
bash dataset/prepare/download_glove.sh
```

## Datasets
We are using two 3D human motion-language dataset: HumanML3D and KIT-ML. For both datasets, you could find the details as well as download [link](https://github.com/EricGuo5513/HumanML3D).
We perturbed the input texts based on the two datasets mentioned. You can access the perturbed text dataset through the following [link]().

### Train

```

```

### Evaluation

```

```

## Acknowledgements



## Citing
If you find this code useful for your research, please consider citing the following paper:

```bibtex

```


