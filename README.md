# SITTA
The repo contains official PyTorch Implementation of the paper [SITTA: Single Image Texture Translation for Data Augmentation](https://link.springer.com/chapter/10.1007/978-3-031-25063-7_1).

European Conference on Computer Vision (ECCV) Workshops, 2022

#### Authors: 
* [Boyi Li](https://sites.google.com/site/boyilics/home)
* [Yin Cui](https://scholar.google.com/citations?hl=zh-CN&user=iP5m52IAAAAJ)
* [Tsung-Yi Lin](https://scholar.google.com/citations?hl=zh-CN&user=_BPdgV0AAAAJ)
* [Serge Belongie](https://scholar.google.com/citations?user=ORr4XJYAAAAJ&hl=zh-CN)



### Overview

Recent advances in image synthesis enables one to translate images by learning the mapping between a source domain and a target domain. Existing methods tend to learn the distributions by training a model on a variety of datasets, with results evaluated largely in a subjective manner. Relatively few works in this area, however, study the potential use of semantic image translation methods for image recognition tasks. In this paper, we explore the use of Single Image Texture Translation (SITT) for data augmentation. We first propose a lightweight model for translating texture to images based on a single input of source texture, allowing for fast training and testing. Based on SITT, we then explore the use of augmented data in long-tailed and few-shot image classification tasks. We find the proposed method is capable of translating input data into a target domain, leading to consistent improved image recognition performance. Finally, we examine how SITT and related image translation methods can provide a basis for a data-efficient, augmentation engineering approach to model training.

## Usage
### Environment
CUDA 10.1, pytorch 1.3.1

### Dataset Preparation

<table>
  <thead>
    <tr style="text-align: right;">
       <th></th>
      <th>dataset</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
       <th>0</th>
       <td>SITT leaves images from <a href="https://arxiv.org/abs/2004.11958">Plant Pathology 2020</a> </td>
      <td><a href="https://drive.google.com/drive/folders/1GOmB86w-uVaKo5EydA0YgspPxkE0grbt?usp=sharing">download</a></td>
    </tr>
  </tbody>
</table>

### Running 
`bash run.sh`

If you find this repo useful, please cite:
```
@InProceedings{10.1007/978-3-031-25063-7_1,
author="Li, Boyi
and Cui, Yin
and Lin, Tsung-Yi
and Belongie, Serge",
editor="Karlinsky, Leonid
and Michaeli, Tomer
and Nishino, Ko",
title="SITTA: Single Image Texture Translation for Data Augmentation",
booktitle="Computer Vision -- ECCV 2022 Workshops",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="3--20",
abstract="Recent advances in data augmentation enable one to translate images by learning the mapping between a source domain and a target domain. Existing methods tend to learn the distributions by training a model on a variety of datasets, with results evaluated largely in a subjective manner. Relatively few works in this area, however, study the potential use of image synthesis methods for recognition tasks. In this paper, we propose and explore the problem of image translation for data augmentation. We first propose a lightweight yet efficient model for translating texture to augment images based on a single input of source texture, allowing for fast training and testing, referred to as Single Image Texture Translation for data Augmentation (SITTA). Then we explore the use of augmented data in long-tailed and few-shot image classification tasks. We find the proposed augmentation method and workflow is capable of translating the texture of input data into a target domain, leading to consistently improved image recognition performance. Finally, we examine how SITTA and related image translation methods can provide a basis for a data-efficient, ``augmentation engineering'' approach to model training.",
isbn="978-3-031-25063-7"
}
```

