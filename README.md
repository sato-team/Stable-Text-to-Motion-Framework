# SATO: Stable Text-to-Motion Framework

[Wenshuo chen](https://github.com/shurdy123), [Hongru Xiao](https://github.com/Hongru0306), [Erhang Zhang](https://github.com/zhangerhang), [Lijie Hu](https://sites.google.com/view/lijiehu/homepage), [Lei Wang](https://leiwangr.github.io/), [Mengyuan Liu](), [Chen Chen](https://www.crcv.ucf.edu/chenchen/)

[![Website shields.io](https://img.shields.io/website?url=http%3A//poco.is.tue.mpg.de)]() [![YouTube Badge](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)]()  [![arXiv](https://img.shields.io/badge/arXiv-2308.12965-00ff00.svg)]()  

<!-- <div style="display:flex;">
    <img src="assets/run_lola.gif" width="45%" style="margin-right: 1%;">
    <img src="assets/yt_solo.gif" width="45%">
</div> -->

<p align="center">
<table align="center">
  <tr>
    <th colspan="4">Original text: Walking forward in an even pace.</th>
  </tr>
  <tr>
    <th align="center"><u><a href="https://github.com/Mael-zys/T2M-GPT"><nobr>T2M-GPT</nobr> </a></u></th>
    <th align="center"><u><a href="https://guytevet.github.io/mdm-page/"><nobr>MDM</nobr> </a></u></th>
    <th align="center"><u><a href="https://github.com/EricGuo5513/momask-codes"><nobr>MoMask</nobr> </a></u></th>
    <th align="center"><u><a href="https://github.com/sato-team/Stable-Text-to-motion-Framework"><nobr>SATO</nobr> </a></u></th>
  </tr>

  <tr>
    <td width="250" align="center"><img src="images/example/walk/gpt.gif" width="150px" height="150px" alt="gif"></td>
    <td width="250" align="center"><img src="images/example/walk/mdm.gif" width="150px" height="150px" alt="gif"></td>
    <td width="250" align="center"><img src="images/example/walk/momask.gif" width="150px" height="150px" alt="gif"></td>
    <td width="250" align="center"><img src="images/example/walk/sato.gif" width="150px" height="150px" alt="gif"></td>
  </tr>

  <tr>
    <th colspan="4" >Perturbed text: <span style="color: red;">Going ahead</span> in an even pace.</th>
  </tr>
  <tr>
    <th align="center"><u><a href="https://github.com/Mael-zys/T2M-GPT"><nobr>T2M-GPT</nobr> </a></u></th>
    <th align="center"><u><a href="https://guytevet.github.io/mdm-page/"><nobr>MDM</nobr> </a></u></th>
    <th align="center"><u><a href="https://github.com/EricGuo5513/momask-codes"><nobr>MoMask</nobr> </a></u></th>
    <th align="center"><nobr>SATO</nobr> </th>
  </tr>

  <tr>
    <td width="250" align="center"><img src="images/example/go/gpt.gif" width="150px" height="150px" alt="gif"></td>
    <td width="250" align="center"><img src="images/example/go/mdm.gif" width="150px" height="150px" alt="gif"></td>
    <td width="250" align="center"><img src="images/example/go/momask.gif" width="150px" height="150px" alt="gif"></td>
    <td width="250" align="center"><img src="images/example/go/sato.gif" width="150px" height="150px" alt="gif"></td>
  </tr>
</table>
</p>


## Table of Content

* [1. Setup and Installation](#setup)

* [2.Dependencies](#Dependencies)

* [3. Quick Start](#quickstart)

* [4. Datasets](#datasets)

* [4. Train](#train)

* [5. Evaluation](#eval)

* [6. Acknowledgments](#acknowledgements)

  

## Setup and Installation <a name="setup"></a>

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

## Dependencies<a name="Dependencies"></a>

```shell
bash dataset/prepare/download_extractor.sh
bash dataset/prepare/download_glove.sh
```

## **Quick Start**<a name="quickstart"></a>

A quick reference guide for using our code is provided in quickstart.ipynb.

## Datasets<a name="datasets"></a>

We are using two 3D human motion-language dataset: HumanML3D and KIT-ML. For both datasets, you could find the details as well as download [link](https://github.com/EricGuo5513/HumanML3D).
We perturbed the input texts based on the two datasets mentioned. You can access the perturbed text dataset through the following [link](https://drive.google.com/file/d/1TZx-mDGeRXyuCsfuz8_oJhjiye6FWsTh/view?usp=sharing).

### **Train**<a name="train"></a>

We will release the training code soon.

### **Evaluation**<a name="eval"></a>

You can download the pretrained models in this [link](https://drive.google.com/drive/folders/1rs8QPJ3UPzLW4H3vWAAX9hJn4ln7m_ts?usp=sharing). 

```shell
python eval_t2m.py --resume-pth pretrained/net_best_fid.pth --clip_path pretrained/clip_best_fid.pth
```

## Acknowledgements<a name="acknowledgements"></a>

We appreciate helps from :

- Open Source Code：[T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MoMask ](https://github.com/EricGuo5513/momask-codes)etc.
- [Wenshuo chen](https://github.com/shurdy123), [Hongru Xiao](https://github.com/Hongru0306), [Erhang Zhang](https://github.com/zhangerhang), [Lijie Hu](https://sites.google.com/view/lijiehu/homepage), [Lei Wang](https://leiwangr.github.io/), [Mengyuan Liu](), [Chen Chen](https://www.crcv.ucf.edu/chenchen/) for discussions and guidance throughout the project, which has been instrumental to our work.
- [Zhen Zhao](https://github.com/Zanebla) for project website.

## Citing<a name="citing"></a>

If you find this code useful for your research, please consider citing the following paper:

```bibtex

```

