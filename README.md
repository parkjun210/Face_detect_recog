# SNU: Face detection & recognition

# Environments
```
conda create -n ENV_NAME python=3.7

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

pip install opencv-python

pip install tqdm
```


# 코드 실행 가이드 라인

## === 학습된 ckpt ===

**Face recognition**

아래 링크에서 미리 학습한 weight(jun_adaface/Backbone_Best.pth)를 다운 받아 "SNU_FaceRecognition/ckpt" 폴더 생성 후 그 안에 배치한다.

Ex. "SNU_FaceRecognition/ckpt/Backbone_Best.pth"

구글 드라이브 주소 : https://drive.google.com/drive/folders/1zjp58d4T3vB6UA6ReEgzdemOoyxTZPbP?usp=sharing

**Face detection**

아래 링크에서 미리 학습한 ckpt 폴더(resnet_anc2_casT_fpn3)를 다운 받아 "Snu_FaceDetection/experiments" 폴더 생성 후 그 안에 배치한다.

구글 드라이브 주소 : https://drive.google.com/drive/folders/1bbxIfmmlhs33uBkTasL6ksnPfabFFpNI?usp=sharing

(2024년 업데이트 weight) 

미리 학습한 ckpt 폴더(tina_iou_anc3_casT_fpn3)를 다운 받아 "Snu_FaceDetection/experiments" 폴더 생성 후 그 안에 배치한다.

구글 드라이브 주소: https://drive.google.com/drive/folders/11ZVjvwctmiO9bPbnVqmn97GX8gGJj_01?usp=sharing


## === Image Inference ===

1. 학습된 face detection ckpt, face recognition ckpt 다운로드 및 detect.cfg 수정
2. library 폴더 안에 face database 폴더 필요 (detect.cfg파일 recognition_library_path 변수 설정)
3. data 폴더 안에 inference할 image가 담긴 폴더 필요 (detect.py의 img_dir 폴더에 inference할 image path 설정)
