# SNU_FaceDetection

# Reference
### paper:
[RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild] (CVPR 2020)

https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html

    @inproceedings{deng2020retinaface,
      title={Retinaface: Single-shot multi-level face localisation in the wild},
      author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={5203--5212},
      year={2020}
    }
    
### code: 
아래 github 의 아키텍쳐를 참고하여 multi-stage로 재현함

https://github.com/biubug6/Pytorch_Retinaface

# 실행 결과 예시 
<img src="https://user-images.githubusercontent.com/57519896/160752508-9d09c9d8-c592-497b-ae56-45ccf2cdd97a.jpg" width="512" height="343"/>
<img src="https://user-images.githubusercontent.com/57519896/160752511-81d432ed-0d88-4683-a158-ee1e70dbaa57.jpg" width="512" height="384"/>

# Environments
```
conda create -n ENV_NAME python=3.7

pip install mediapipe
pip install torch==1.7.0
pip install torchvision==0.8.1
pip install opencv-python
```


# Dataset 다운 주소
train/val/test dataset - widerface

http://shuoyang1213.me/WIDERFACE/


# Directory 설명
    |── data
        ├──> sample_widerface 
             ├──> images : widerface validation set에서 뽑은 10장의 샘플
        ├──> widerface
            ├──> train
                ├──> images : 학습 이미지가 저장되야 하는 폴더
                └──> label.txt : 적절한 입력 포멧으로 변형한 학습 레이블
            ├──> val
                ├──> images : 검증 이미지가 저장되야 하는 폴더
                └──> label.txt : 적절한 입력 포멧으로 변형한 검증 레이블
    |── inference_results
        ├──> result_images : widerface sample 이미지에 대한 실행 결과 이미지 (bbox, confidence 포함)
        └──> resnet_anc2_casT_fpn3_inference_results.txt : widerface sample 이미지에 대한 실행 결과 (bbox, confidence를 저장)
    |── layers
        ├──> multibox_loss.py : face bbox, label, landmarks을 한번에 처리하는 loss 모듈
        └──> prior_box.py : prior box를 생성하는 모듈
    |── models
        ├──> net.py : retinaface 아키텍쳐에 사용되는 모듈 (SSH(=CHM), FPN)
        └──> retinaface.py : retinaface 전체 아키텍쳐  
    |── utils : 다양한 기타 사용 함수들 폴더
    |── config.py : 입력 argument를 관리하는 파일
    |── inference.py : inference용 코드 (GT label이 없을 경우 테스트)
    |── retinaface.yml : 가상환경 파일
    |── test.py : test용 코드(GT label이 있을 경우 테스트)
    └── train.py : train용 코드




# 코드 실행 가이드 라인

## === Train ===
학습용 코드 - train.py

### 1) dataset 준비
   위 dataset 다운 주소를 참고하여 widerface train, validation dataset을 다운받고, directory 설명을 참고하여 train, val 이미지 폴더를 배치한다
   
### 2) 실행

   아래 명령어를 통해 실행한다
   
   python train.py --gpu_num={사용할 gpu index, int} --experiment_name={학습결과를 저장할 폴더 이름, string}
     EX. python train.py --gpu_num=0 --experiment_name='resnet_anc2_casT_fpn3'
   
   기본 epoch는 8000, batch size는 16으로 되어있으며, 변경하고 싶을 시 아래와 같이 추가한다
   
   python train.py --gpu_num=0 --experiment_name='resnet_anc2_casT_fpn3' --epochs={epoch_num} --batch_sixe={batch_size}
   
### 3) 결과 저장
   학습이 종료되면 experiments/ 폴더가 아래와 같이 생성된다
               
        |── experiments
           ├──> {experiment_name}
                ├──> log : 학습과정의 log 파일 (학습 실패 시에도 본 파일 참고)
                └──> ckpt : 학습과정 중의 가장 결과가 좋은 체크포인트 파일 저장
           



## === 학습된 ckpt ===

혹은 아래 링크에서 미리 학습한 ckpt 파일(resnet_anc2_casT_fpn3)을 다운 받아 experiments 폴더를 생성한 후 그 안에 배치한다. 

구글 드라이브 주소 : https://drive.google.com/drive/folders/1bbxIfmmlhs33uBkTasL6ksnPfabFFpNI?usp=sharing


## === Test ===
GT label이 존재하는 dataset에 대해서는 아래 코드를 통해 테스트를 진행한다

테스트용 코드 - test.py (GT 존재해서 AP 측정 가능할 때)

### 1) 데이터셋 확인
   ./data/widerface/val/images 내에 있는 폴더에 대해 테스트를 진행한다

### 2) 코드 실행
   아래 명령어를 통해 실행한다. 
 
   python test.py --gpu_num={사용할 gpu index, int} --experiment_name={테스트에 사용할 ckpt 폴더가 저장된 폴더}
   
    python test.py --gpu_num=0 --experiment_name='resnet_anc2_casT_fpn3'
    
### 3) 결과 저장
   10장 단위로 테스트 진행 과정을 출력하며, 테스트가 종료되면 테스트에 걸린 시간과 AP 결과를  ./experiments/{exp_name}/results/results.txt에 저장한다


## === Inference ===

GT label이 존재하지 않는 dataset에 대해서는 아래 코드를 통해 테스트를 진행한다

테스트용 코드2 - inference.py (GT 존재하지 않아서 AP 측정 불가능)

### 1) dataset
   ./data/{dataset이름}/images/ 폴더를 만들어 inference용 이미지를 넣는다

### 2) 코드 실행

   아래 명령어를 통해 테스트를 실행한다. 
 
   python inference.py 
   
   --gpu_num={사용할 gpu index, int} 
   
   --experiment_name={테스트에 사용할 ckpt 폴더가 저장된 폴더} 
   
   --inference_dir={inference용 이미지가 저장된 폴더, default='sample_widerface/images/'}
   
   --infer_imsize_same={inference용 이미지들의 크기가 일정한지 여부, default=True}
   
   --save_img={inference 결과 이미지를 저장할 지 여부, defalut=False}
   
   --inference_save_folder={결과 이미지를 저장할 폴더 이름, default='inference_results'}
   
   
    python inference.py --gpu_num=0 --inference_dir='sample_widerface/images/' --save_img=True --inference_save_folder='inference_results/'
    
    
### 3) 결과 저장
   10장 단위로 테스트 진행과정을 출력하며, test가 종료된 후에는 ./inference_results 폴더에 결과가 저장된다.

   **주의 : --inference_save_folder를 지정하지 않고 실행 시 덮어씌워질 수 있음
   
        |── inference_results
           ├──> result_images: --save_img=True를 줬을 시 inference 이미지를 저장
           └──> exp_name_inference_results.txt: image 이름과 그 bbox, 신뢰도 결과값을 결과로 저장. 
           

### 4) mask 추출

   source code: https://google.github.io/mediapipe/solutions/selfie_segmentation.html
   
    python inference.py --gpu_num=0 --inference_dir='sample_mask/images/' --inference_save_folder='mask_results/' --mask True --save_mask True
    
   
   10장 단위로 테스트 진행과정을 출력하며, test가 종료된 후에는 --inference_save_folder로 지정한 폴더에 아래 결과가 저장된다.

   **주의 : --inference_save_folder를 지정하지 않고 실행 시 덮어씌워질 수 있음
   
        |── mask_results
           ├──> masks:{기존 image_name + face idx}  face mask(tight한 mask) 이미지가 저장 됨}
           ├──> faces:{기존 image_name + face idx} 얼굴 이미지가 저장 됨}
           ├──> head_masks:{기존 image_name + face idx}  head mask 이미지가 저장 됨}
           └──> exp_name_inference_results.txt: image 이름과 detection bbox, 신뢰도 결과값을 결과로 저장. 
           
   
   코드 상 return하는 값은 inference.py L 282를 참고하여 아래와 같다
   
   * head = face를 정중앙으로 하여 face보다 h, w 모두 4배 크게 잡은 영역 (mask segmentation의 input으로 들어감) 
   
    result_bboxes, face_bboxes, face_masks, head_bboxes, head_masks = result
   
   result_bboxes: 전체 이미지에서 face bbox의 좌표 [x1, y1, x2, y2]
   
   face_bboxes: head 이미지에서 face bbox의 좌표 [x1, y1, x2, y2]
   
   face_masks: face의 mask 이미지 [H, W, C]
   
   head_bboxes: 전체 이미지에서 head bbox의 좌표 [x1, y1, x2, y2]
   
   head_masks: head의 mask 이미지 [H, W, C]
   
   #### 비고: 인물들이 많이 겹쳐져 있을 수록 mask segmentation의 성능이 저하 됨
   
   #### 결과 예시
   
   input 이미지
   
   <img width="400" alt="스크린샷 2022-07-26 오후 6 50 59" src="https://user-images.githubusercontent.com/57519896/180977996-036d43e6-9d89-4888-ae75-b56b3622c2c6.png">
   
   face detection 결과 
   
   <img width="400" src="https://user-images.githubusercontent.com/57519896/180978262-771274ec-ed7d-4443-9047-33bd25ea4e2e.png">
   
   head 이미지
   
   ![48018_origin_49_0](https://user-images.githubusercontent.com/57519896/180978750-adb03bf8-3033-4211-adcf-9c8f7746d657.png)
   
   head_mask 이미지 
   
   ![48018_origin_49_0](https://user-images.githubusercontent.com/57519896/180978796-9538e26c-a889-43fc-b241-872b7ec0ec20.png)
   
   face 이미지
   
   ![48018_origin_49_0](https://user-images.githubusercontent.com/57519896/180978829-b9969d6b-bcbf-4948-8ff0-abb51d7bdfeb.png)
   
   face_mask 이미지 
   
   ![48018_origin_49_0](https://user-images.githubusercontent.com/57519896/180978864-4117262c-53af-4eec-b414-d6c31d3f37a6.png)

  
