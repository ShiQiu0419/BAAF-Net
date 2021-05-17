# Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion (CVPR 2021)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semantic-segmentation-for-real-point-cloud/semantic-segmentation-on-s3dis)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis?p=semantic-segmentation-for-real-point-cloud)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semantic-segmentation-for-real-point-cloud/semantic-segmentation-on-s3dis-area5)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis-area5?p=semantic-segmentation-for-real-point-cloud)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semantic-segmentation-for-real-point-cloud/semantic-segmentation-on-semantic3d)](https://paperswithcode.com/sota/semantic-segmentation-on-semantic3d?p=semantic-segmentation-for-real-point-cloud)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semantic-segmentation-for-real-point-cloud/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=semantic-segmentation-for-real-point-cloud)

This repository is for BAAF-Net introduced in the following paper:
 
"Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion"  
[Shi Qiu](https://shiqiu0419.github.io/), [Saeed Anwar](https://saeed-anwar.github.io/), [Nick Barnes](http://users.cecs.anu.edu.au/~nmb/)  
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2021)

## Paper and Citation
The paper can be downloaded from [here (arXiv)](https://arxiv.org/abs/2103.07074).  
If you find our paper/codes/results are useful, please cite:

    @inproceedings{qiu2021semantic,
      title={Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion},
      author={Qiu, Shi and Anwar, Saeed and Barnes, Nick},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2021},
    }

## Experimental Environment



## Updates
* **04/05/2021** Results for S3DIS dataset (mIoU: **72.2%**, OA: **88.9%**, mAcc: **83.1%**) are [available](https://drive.google.com/file/d/1GnHhfeItJDJCM0rIFLR5H7SrWRwO37Y4/view?usp=sharing) now. 
* **04/05/2021** Test results (sequence 11-21: mIoU: **59.9%**, OA: **89.8%**) for SemanticKITTI dataset are [available](https://drive.google.com/file/d/1FkyNfknwnJ2YnwUvPhMQGvJXCW--mqkK/view?usp=sharing) now.
* **04/05/2021** Validation results (sequence 08: mIoU: **58.7%**, OA: **91.3%**) for SemanticKITTI are [available](https://drive.google.com/file/d/1grQ57rZXL34mAOmI_3IASovu_APOPMI3/view?usp=sharing) now.
* To be continued.

## Results on S3DIS Dataset
<p align="center">
  <img width="1000" src="https://github.com/ShiQiu0419/BAAF-Net/blob/main/s3dis.png">
</p>

* Download our results (**ply** files) via [google drive](https://drive.google.com/file/d/1GnHhfeItJDJCM0rIFLR5H7SrWRwO37Y4/view?usp=sharing) for visualizations/comparisons.
* Codes for 6-fold cross-validation can be found from [here](https://github.com/QingyongHu/RandLA-Net/blob/master/utils/6_fold_cv.py).
* More Functions about loading/writing/etc. **ply** files can be found from [here](https://github.com/QingyongHu/RandLA-Net/blob/master/utils/6_fold_cv.py).

## Results on SemanticKITTI Dataset
* Online test results (sequence 11-21): mIoU: **59.9%**, OA: **89.8%**
* Download our **test** results (sequence 11-21 **label** files) via [google drive](https://drive.google.com/file/d/1FkyNfknwnJ2YnwUvPhMQGvJXCW--mqkK/view?usp=sharing) for visualizations/comparisons.

<p align="center">
  <img width="1200" src="https://github.com/ShiQiu0419/BAAF-Net/blob/main/kitti_08.png">
</p>  

* Validation results (sequence 08): mIoU: **58.7%**, OA: **91.3%**
* Download our **validation** results (sequence 08 **label** files) via [google drive](https://drive.google.com/file/d/1grQ57rZXL34mAOmI_3IASovu_APOPMI3/view?usp=sharing) for visualizations/comparisons.
* Visualization tools can be found from [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api).

## Acknowledgment
The code is built on [RandLA-Net](https://github.com/QingyongHu/RandLA-Net). We thank the authors for sharing the codes.
