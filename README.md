# Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion (CVPR 2021)
This repository is for BAAF-Net introduced in the following paper:
 
"Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion"  
[Shi Qiu](https://shiqiu0419.github.io/), [Saeed Anwar](https://saeed-anwar.github.io/), [Nick Barnes](http://users.cecs.anu.edu.au/~nmb/)  
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2021)

## Paper
The paper can be downloaded from [here (arXiv)](https://arxiv.org/abs/2103.07074).

## Updates
* **04/05/2021** Results for S3DIS dataset (mIoU: **72.2%**, OA: **88.9%**, mAcc: **83.1%**) are available at [google drive](https://drive.google.com/file/d/1GnHhfeItJDJCM0rIFLR5H7SrWRwO37Y4/view?usp=sharing) now. 
* **04/05/2021** [Test](https://drive.google.com/file/d/1FkyNfknwnJ2YnwUvPhMQGvJXCW--mqkK/view?usp=sharing) (mIoU: **59.9%**, OA: **89.8%**) and [validation](https://drive.google.com/file/d/1grQ57rZXL34mAOmI_3IASovu_APOPMI3/view?usp=sharing) (**58.7%**, OA: **91.3%**) results for SemanticKITTI dataset are available now.
* To be continued.

## Results on S3DIS Dataset
<p align="center">
  <img width="1200" src="https://github.com/ShiQiu0419/BAAF-Net/blob/main/s3dis.png">
</p>

* Download our results (**ply** files) via [google drive](https://drive.google.com/file/d/1GnHhfeItJDJCM0rIFLR5H7SrWRwO37Y4/view?usp=sharing) for visualizations/comparisons.
* Codes for 6-fold cross-validation can be found from [here](https://github.com/QingyongHu/RandLA-Net/blob/master/utils/6_fold_cv.py).
* More Functions about loading/writing/etc. **ply** files can be found from [here](https://github.com/QingyongHu/RandLA-Net/blob/master/utils/6_fold_cv.py).

## Results on SemanticKITTI Dataset
* Online test results (sequence 11-21): mIoU: **59.9%**, OA: **89.8%**
* Download our **test** results (sequence 11-21 **label** files) via [google drive](https://drive.google.com/file/d/1FkyNfknwnJ2YnwUvPhMQGvJXCW--mqkK/view?usp=sharing) for visualizations/comparisons.
* Validation results (sequence 08): mIoU: **58.7%**, OA: **91.3%**
* Download our **validation** results (sequence 08 **label** files) via [google drive](https://drive.google.com/file/d/1grQ57rZXL34mAOmI_3IASovu_APOPMI3/view?usp=sharing) for visualizations/comparisons.
* Visualization tools can be found from [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api).

## Code
Coming soon.

## Citation

If you find our paper/results are useful, please cite:

    @inproceedings{qiu2021semantic,
      title={Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion},
      author={Qiu, Shi and Anwar, Saeed and Barnes, Nick},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2021},
    }
