# Kaggle Competition - Sartorius
---


# Team name/members
- **Team name :** SegMan
- **Members**
  - [Park Hyunyong](https://github.com/hyunyongPark)
  - [Shin younghoon](https://github.com/Yphy)

<br/><br/><br/>
# Introduction
![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/sartorius_title.PNG?raw=true)

Kaggle 주최인 [instance segmenatation대회](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/overview)로서, 국제 제약 및 실험실 장비 공급업체인 Sartorius AG가 데이터를 제공하였다. 
기존 8개의 종류의 신경세포를 감지하는 시스템을 실험하고 있었으나, 그 중 SH-SY5Y라는 신경세포종류가 낮은 정밀도 및 정확도를 나타내서 복병이었다고 한다. 
따라서 본 대회에서는 SH-SY5Y, CORT, ASTRO 세 가지 신경세포유형에 대한 instance를 감지해내는 task를 요구한다. 

<br/><br/><br/>
# Dataset / Evaluation

### 1) Dataset
![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/img_example.PNG?raw=true)

본 데이터셋은 위의 그림과 같이 3가지의 Class를 가지고 있으며, 좌표값과 class정보 그리고 여러 메타정보를 가지고 있는 csv file과 각 class별 이미지 file들이 존재한다. 

csv file의 각 column은 다음과 같다.
> * id - unique identifier for object
> * annotation - run length encoded pixels for the identified neuronal cell
> * width - source image width
> * height - source image height
> * cell_type - the cell line
> * plate_time - time plate was created
> * sample_date - date sample was created
> * sample_id - sample identifier
> * elapsed_timedelta - time since first image taken of sample

여기서 알아야할 것은 RLE Encoding 방식의 mask정보이다.
RLE는 아래의 그림을 참고하도록 한다. 
![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/rle_encode.png?raw=true)

위와 같이 해당 데이터는 rle encoded value로 구성된 mask정보들이 존재하며 이들은 각 이미지에 해당하는 세포 instance들이라고 볼 수 있다. 

추가로 LiveCell Dataset도 존재하는데, 이는 기존 8개의 class에 대한 인스턴스 정보와 현미경 세포 이미지들의 데이터들로 구성되어 있다. 

### 2) Evaluation
![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/evaluation_1.PNG?raw=true)

본 대회에서는 서로다른 IoU threshold값들에 대한 평균정밀도(AP)를 계산하는 식을 사용한다. 
즉, threshold 0.5~0.95까지의 iou임계값들을 사용하며 이를 다시 평균하여 사용하는 MaP(Mean Average Precision)로 평가를 한다.

![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/evaluation_2.PNG?raw=true)

<br/><br/><br/>
# Solution
### 1) modeling
본 대회에서 우리팀은 Facebook에서 개발한 [Detectron2](https://github.com/facebookresearch/detectron2) 라이브러리를 사용하여 진행하였다.
Detectron2은 이전 프로젝트에서 사용한 이력이 있어 어색함은 덜했으며, 여러가지 pre-trained모델과 지원하는 parameter에 대한 튜닝을 위주로 진행하였다. 
초기 사용모델은 Mask R-CNN(ResNet+FPN)으로 실험하였으며, 최종적으로는 Cascade Mask R-CNN(ResNet+FPN)모델을 채택하여 진행하였다. 
Single Model 추론 파이프라인 및 프레임워크는 다음과 같다. 

![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/casacde_infer_frmwork.PNG?raw=true)

학습은 pre-trained model을 기반으로 fine-tune하는 방식을 거쳤는데 이는 다음의 파이프라인을 참고하면 된다. 

![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/training_pipeline_.PNG?raw=true)

### 2) Ensemble
위에서 학습된 모델들은 Ensemble을 사용하여 진행한다.

Ensemble 기법으로는 각 모델에서 예측되는 binary mask를 합집합으로 나열하고, class score 및 bbox 값들에 대해 NMS(Non-Maximum Suppression)를 진행하게 된다. 
여기서 NMS에 필수적인 threshold값이 존재하기 때문에, 이 파라미터를 실험하는데 많은 시간을 소요하였다. 

다음의 파이프라인을 참고하면 된다.

![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/ensemble_pipeline.PNG?raw=true)



<br/><br/><br/>
# Conclusion
최종 78등(Top 6%)로 동메달을 획득하였다. 아쉽게도 77등까지가 은메달이었지만 그 범위에 한 등수차이로 도달하지 못하였다.

![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/final_rank.PNG?raw=true)

대회 진행을 하면서 아쉬운 점은 다음과 같다.
### 1) Training a detection model for each class
> 각 class별로 인스턴스를 나누어 학습하는 방식이 상위 solution에 있었다. 각 이미지 한장당 multi class가 존재하지않고 동일한 class만 존재하기 때문에 semantic segmentation의 관점으로 접근한 것으로 보인다. 따라서 각 class별 모델을 학습하면 조금 더 mask를 예측하는데에 집중할 수 있을 것으로 보여진다. 

### 2) Use of the Unet
> 상위 solution에서 Semantic segmentation에서 효과적인 Unet을 학습하여 앙상블하는데에 활용한 것을 볼 수 있다. semantic seg + instance seg의 효과를 기대할 수 있을 것으로 보여진다.

### 3) WBF Ensemble method
> object detection 분양에서 쓰이는 앙상블 기법으로 WBF(Weighted Boxes Fusion) 방법이 있다. 이 방법은 여러 bounding box를 효과적으로 앙상블하는 방법인데, 이런 방법에 대한 리서치를 못한 것이 매우 아쉬움이 느껴진다. 
