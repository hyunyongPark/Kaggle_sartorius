# Kaggle_sartorius
---
# Introduction
![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/sartorius_title.PNG?raw=true)

Kaggle 주최인 [instance segmenatation대회](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/overview)로서, 국제 제약 및 실험실 장비 공급업체인 Sartorius AG가 데이터를 제공하였다. 
기존 8개의 종류의 신경세포를 감지하는 시스템을 실험하고 있었으나, 그 중 SH-SY5Y라는 신경세포종류가 낮은 정밀도 및 정확도를 나타내서 복병이었다고 한다. 
따라서 본 대회에서는 SH-SY5Y, CORT, ASTRO 세 가지 신경세포유형에 대한 instance를 감지해내는 task를 요구한다. 

---
# Dataset / Evaluation

### 1) Dataset
![image](https://github.com/hyunyongPark/Kaggle_sartorius/blob/master/img/img_example.PNG?raw=true)

본 데이터셋은 위의 그림과 같이 3가지의 Class를 가지고 있으며, 좌표값과 class정보 그리고 여러 메타정보를 가지고 있는 csv file과 각 class별 이미지 file들이 존재한다. 

csv file의 각 column은 다음과 같다.
>>> * id - unique identifier for object
>>> * annotation - run length encoded pixels for the identified neuronal cell
>>> * width - source image width
>>> * height - source image height
>>> * cell_type - the cell line
>>> * plate_time - time plate was created
>>> * sample_date - date sample was created
>>> * sample_id - sample identifier
>>> * elapsed_timedelta - time since first image taken of sample


