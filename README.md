# Kaggle Competition - Sartorius
---



# Team name/members
- Team name : SegMan
- Members
  - [Park Hyunyong](https://github.com/hyunyongPark)
  - [Shin younghoon](https://github.com/Yphy)



















# Introduction
![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/sartorius_title.PNG?raw=true)

Kaggle 주최인 [instance segmenatation대회](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/overview)로서, 국제 제약 및 실험실 장비 공급업체인 Sartorius AG가 데이터를 제공하였다. 
기존 8개의 종류의 신경세포를 감지하는 시스템을 실험하고 있었으나, 그 중 SH-SY5Y라는 신경세포종류가 낮은 정밀도 및 정확도를 나타내서 복병이었다고 한다. 
따라서 본 대회에서는 SH-SY5Y, CORT, ASTRO 세 가지 신경세포유형에 대한 instance를 감지해내는 task를 요구한다. 



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

### 2) Evaluation
![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/evaluation_1.PNG?raw=true)

본 대회에서는 서로다른 IoU threshold값들에 대한 평균정밀도(AP)를 계산하는 식을 사용한다. 
즉, threshold 0.5~0.95까지의 iou임계값들을 사용하며 이를 다시 평균하여 사용하는 MaP(Mean Average Precision)로 평가를 한다.

![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/evaluation_2.PNG?raw=true)




# Solution
본 대회에서 우리팀은 
