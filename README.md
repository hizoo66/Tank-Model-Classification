# 🛡️ Tank Detection & Classification Project

본 프로젝트는 **군용 전차(Tank) 탐지 및 분류를 목표로 한 컴퓨터 비전 프로젝트**로,
YOLO 기반 객체 탐지 이후 **CNN 분류 모델(ResNet, EfficientNet)을 활용한 2단계 분류 구조**를 설계하고,
모델별 성능 비교를 수행합니다.

---

## 📌 Project Overview

* **목표**

  * 이미지 내 전차(Tank) 및 비전차 차량 탐지
  * 탐지된 객체를 기반으로 **정확한 클래스 분류**
  * YOLO 단일 모델과 **YOLO + CNN 분류 구조의 성능 비교**

* **접근 방식**

  1. YOLO를 이용한 객체 탐지 및 1차 분류
  2. YOLO 탐지 결과 기반 이미지 Crop
  3. Crop된 이미지를 CNN 분류 모델에 입력
  4. 모델별 성능 비교 분석

---

## 🗂️ Dataset

### 1️⃣ Tank / Non-Tank Dataset

#### 🔹 Tank 이미지 데이터

* 형식: `.jpg`, `.txt` (YOLO format)
* 데이터 수: **1,785장**
* 라벨링: O
* 출처:
  [https://www.kaggle.com/datasets/saifkjarallah/tank-yolo-format-annotation-and-class-label5](https://www.kaggle.com/datasets/saifkjarallah/tank-yolo-format-annotation-and-class-label5)

#### 🔹 Non-Tank (Vehicle) 이미지 데이터

* 형식: `.jpg`, `.txt` (YOLO format)
* 데이터 수: **3,000장**
* 라벨링: O
* 출처:
  [https://www.kaggle.com/datasets/nadinpethiyagoda/vehicle-dataset-for-yolo](https://www.kaggle.com/datasets/nadinpethiyagoda/vehicle-dataset-for-yolo)

---

## 🏷️ Classes

```python
CLASS_NAMES = {
    0: "tank",
    1: "car",
    2: "threewheel",
    3: "bus",
    4: "truck",
    5: "motorbike",
    6: "van"
}
```

* 총 **7개 클래스**
* 군용 전차 1종 + 민간 차량 6종

---

## 🧠 Model Architecture

### 1️⃣ YOLO-based Object Detection

* 역할

  * 이미지 내 객체 위치 탐지 (Bounding Box)
  * 1차 클래스 분류
* 입력: 원본 이미지
* 출력:

  * Bounding Box
  * Class ID
  * Confidence Score

---

### 2️⃣ Crop-based Image Classification

YOLO 라벨 데이터를 기반으로 객체 영역을 Crop하여 **정밀 분류 수행**

#### 🔹 Crop 과정

* YOLO Bounding Box 기준 이미지 Crop
* Resize & Normalize
* CNN 분류 모델 입력

---

### 3️⃣ Classification Models

| Model        | Purpose           |
| ------------ | ----------------- |
| ResNet       | 깊은 네트워크 기반 안정적 분류 |
| EfficientNet | 파라미터 효율적 고성능 분류   |

* Pretrained 모델 활용 (ImageNet)
* Fine-tuning 수행

---

## 🔬 Experiments & Comparison

### 비교 실험 항목

1. **YOLO 단일 모델 분류 성능**
2. **YOLO + ResNet 분류 성능**
3. **YOLO + EfficientNet 분류 성능**

### 평가 지표

* Accuracy
* Precision / Recall
* F1-score
* Confusion Matrix

---

## 📊 Expected Contributions

* 단일 탐지 모델(YOLO) 대비
  👉 **2단계 구조(YOLO + CNN)의 분류 성능 개선 효과 분석**
* 전차 탐지와 민간 차량 오분류 감소
* 실제 감시/국방 영상 분석 시스템에 적용 가능한 구조 제시

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Ultralytics YOLO
* OpenCV
* NumPy
* Matplotlib / Seaborn

---

## 📁 Project Structure (예시)

```
├── tank_datasets/
│   ├── yolo_dataset/
│   ├── cropped_images/
├── models/
│   ├── yolo/
│   ├── resnet/
│   ├── efficientnet/
├── notebooks/
│   ├── tank_nontank_resnet_efficientnet.ipynb
│   ├── tank_nontank_yolo.ipynb
├── utils/
│   ├── crop.py
│   ├── dataset.py
├── README.md
```

---

## 🚀 Future Work

* Few-shot learning 기반 전차 세부 모델 분류 (K1, K2, T-90 등)
* 실시간 영상 스트림 적용
* 탐지 실패 사례 분석 및 데이터 증강 전략 고도화

---

## ✍️ Author

* Tank Detection & Classification Project
* Computer Vision / Deep Learning

---

### 🔑 한 줄 요약 (README 첫 문장용)

> **YOLO 기반 객체 탐지와 CNN 분류 모델을 결합한 전차 탐지 및 차량 분류 성능 비교 프로젝트**

---
