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
The paper can be downloaded from [here (CVF)](https://openaccess.thecvf.com/content/CVPR2021/papers/Qiu_Semantic_Segmentation_for_Real_Point_Cloud_Scenes_via_Bilateral_Augmentation_CVPR_2021_paper.pdf) or [here (arXiv)](https://arxiv.org/abs/2103.07074).  
If you find our paper/codes/results are useful, please cite:

    @inproceedings{qiu2021semantic,
      title={Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion},
      author={Qiu, Shi and Anwar, Saeed and Barnes, Nick},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages={1757-1767},
      year={2021}
    }

## Updates
* **04/05/2021** Results for S3DIS dataset (mIoU: **72.2%**, OA: **88.9%**, mAcc: **83.1%**) are [available](https://drive.google.com/file/d/1GnHhfeItJDJCM0rIFLR5H7SrWRwO37Y4/view?usp=sharing) now. 
* **04/05/2021** Test results (sequence 11-21: mIoU: **59.9%**, OA: **89.8%**) for SemanticKITTI dataset are [available](https://drive.google.com/file/d/1FkyNfknwnJ2YnwUvPhMQGvJXCW--mqkK/view?usp=sharing) now.
* **04/05/2021** Validation results (sequence 08: mIoU: **58.7%**, OA: **91.3%**) for SemanticKITTI are [available](https://drive.google.com/file/d/1grQ57rZXL34mAOmI_3IASovu_APOPMI3/view?usp=sharing) now.
* **28/05/2021** Pretrained models can be downloaded on all 6 areas of S3DIS dataset are available at [google drive](https://drive.google.com/file/d/1DkZeMxJ_ibngwPiW5K0Celcx-nEbaBFb/view?usp=sharing). 
* **28/05/2021** codes released!

## Settings
* The project is tested on Python 3.6, Tensorflow 1.13.1 and cuda 10.0
* Then install the dependencies: ```pip install -r helper_requirements.txt```
* And compile the cuda-based operators: ```sh compile_op.sh```  
(Note: may change the cuda root directory ```CUDA_ROOT``` in ```./util/sampling/compile_ops.sh```)

## Dataset
* Download S3DIS dataset from <a href="https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1">here</a>.
* Unzip and move the folder ```Stanford3dDataset_v1.2_Aligned_Version``` to `./data`.
* Run: ```python utils/data_prepare_s3dis.py```  
(Note: may specify other directory as ```dataset_path``` in ```./util/data_prepare_s3dis.py```)

## Training/Test
* Training:
```
python -B main_S3DIS.py --gpu 0 --mode train --test_area 5
```  
(Note: specify the `--test_area` from `1~6`)
* Test:
```
python -B main_S3DIS.py --gpu 0 --mode test --test_area 5 --model_path 'pretrained/Area5/snap-32251'
```  
(Note: specify the `--test_area` index and the trained model path `--model_path`)

## 6-fold Cross Validation
* Conduct training and test on **each area**.
* Extract **all test results**, `Area_1_conferenceRoom_1.ply` ... `Area_6_pantry_1.ply` (272 `.ply` files in total), to the folder `./data/results`
* Run: `python utils/6_fold_cv.py`  
(Note: may change the target folder `original_data_dir` and the test results `base_dir` in ```./util/6_fold_cv.py```)

## Pretrained Models and Results on S3DIS Dataset
<p align="center">
  <img width="1000" src="https://github.com/ShiQiu0419/BAAF-Net/blob/main/s3dis.png">
</p>

* BAAF-Net pretrained models on all 6 areas can be downloaded from [google drive](https://drive.google.com/file/d/1DkZeMxJ_ibngwPiW5K0Celcx-nEbaBFb/view?usp=sharing).
* Download our results (**ply** files) via [google drive](https://drive.google.com/file/d/1GnHhfeItJDJCM0rIFLR5H7SrWRwO37Y4/view?usp=sharing) for visualizations/comparisons.
* More Functions about loading/writing/etc. **ply** files can be found from [here](https://github.com/ShiQiu0419/BAAF-Net/blob/main/helper_ply.py).

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
