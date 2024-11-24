# SNU: Face detection & recognition

# Reference
### paper:
[RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild] (CVPR 2020)

[TinaFace: Strong but Simple Baseline for Face Detection]

https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html

    @inproceedings{deng2020retinaface,
      title={Retinaface: Single-shot multi-level face localisation in the wild},
      author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={5203--5212},
      year={2020}
    }

    @article{zhu2020tinaface,
      title={Tinaface: Strong but simple baseline for face detection},
      author={Zhu, Yanjia and Cai, Hongxiang and Zhang, Shuhan and Wang, Chenhao and Xiong, Yichao},
      journal={arXiv preprint arXiv:2011.13183},
      year={2020}
    }
    
[AdaFace: Quality Adaptive Margin for Face Recognition] (CVPR 2022)

https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html

@InProceedings{Kim_2022_CVPR,
    author    = {Kim, Minchul and Jain, Anil K. and Liu, Xiaoming},
    title     = {AdaFace: Quality Adaptive Margin for Face Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {18750-18759}
}



# 실행 결과 예시 

![Cast-of-Friends](https://user-images.githubusercontent.com/68048434/182092892-ac65e776-92ca-4a34-8d35-8078c82a77fc.jpg)
![friends_lib](https://user-images.githubusercontent.com/68048434/182092896-e06dfa7a-55d8-4c08-a8f2-4c7c9bab4d34.jpg)


# Environments
```
conda create -n face_api python=3.9

conda activate face_api

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

pip install opencv-python tqdm

pip install numpy==1.23.5

pip install scikit-learn

```


# Directory 설명
    |── SNU_FaceRecognition : Face image recognition 폴더
    |── data                : image & video inference용 input data가 저장되는 폴더
    |── library             : face recognition에 필요한 Library data(aligned face)를 모아놓는 폴더
    |── pipeline_tools
        ├──> recognition.py : detected face image와 library face image를 비교하는 코드
    |── SNU_FaceDetection   : Face detection code 폴더
    |── detect.cfg          : 입력 argument를 관리하는 파일
    |── img_inference.py    : image inference 코드
    └── video_inference.py  : video inference 코드




# 코드 실행 가이드 라인


## === 학습된 ckpt ===


**Face recognition**

아래 링크에서 미리 학습한 weight(jun_adaface/Backbone_Best.pth)를 다운 받아 "SNU_FaceRecognition/ckpt" 폴더 생성 후 그 안에 배치한다.

Ex. "SNU_Face_Detect_Recog/SNU_FaceRecognition/ckpt/Backbone_Best.pth"

구글 드라이브 주소 : https://drive.google.com/drive/folders/1zjp58d4T3vB6UA6ReEgzdemOoyxTZPbP?usp=sharing


**Face detection**

미리 학습한 ckpt 폴더(tina_iou_anc3_casT_fpn3)를 다운 받아 "SNU_FaceDetection/experiments" 폴더 안에 배치한다.

Ex. "SNU_Face_Detect_Recog/SNU_FaceDetection/experiments/tina_iou_anc3_casT_fpn3"

구글 드라이브 주소: https://drive.google.com/drive/folders/11ZVjvwctmiO9bPbnVqmn97GX8gGJj_01?usp=sharing



## === library (face database) ===

aligned face image가 담긴 face database가 들어있는 폴더

Ex.

    |── library
        ├──> class 
            ├──> Brad_Pitt
                ├──> Brad_Pitt_0.jpg
                ├──> Brad_Pitt_1.jpg
            ├──> Courteney_Cox
                ├──> Courteney_Cox.jpg
            ├──> ...
            ├──> David_Schwimmer
                ├──> David_Schwimmer_0.jpg
                ├──> David_Schwimmer_1.jpg
                ├──> David_Schwimmer_2.jpg


## === Image Inference ===

### 1) 필요조건

    1. 학습된 face detection ckpt, face recognition ckpt 필요

    2. face database 필요


### 2) 코드 실행

  아래 명령어를 통해 실행한다. 
 
  python img_inference.py 

  img_detect.cfg 변수 설명
  
    detection_weight_file           = Detection weight file path

    recognition_weight_file         = Recognition weight file path
    
    recognition_library_path        = face database dir path

    input_dir                       = inference할 input dir path
    
    output_dir                      = inference 결과 이미지 저장 dir path
    
    gpu_num                         = 사용할 GPU Device Index

    infer_imsize_same               = Inference dir 안 image 크기가 다르면 False

    detect_save_library             = Face database 생성을 위한 Detected aligned image를 저장 유무
    
    recognition_inference_save_folder = 인식된 최종 결과 이미지 저장 dir path

    recognition_save_img            = 결과 이미지 저장 유무



## === Video Inference ===

  아래 명령어를 통해 실행한다. 
 
  python video_inference.py 

  video_detect.cfg 변수 설명
  
    detection_weight_file           = Detection weight file path

    recognition_weight_file         = Recognition weight file path
    
    recognition_library_path        = face database dir path

    input_video_path                = inference할 video path
    
    output_dir                      = inference 결과 저장 dir path
    
    gpu_num                         = 사용할 GPU Device Index
