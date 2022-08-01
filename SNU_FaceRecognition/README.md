# SNU_FaceRecognition

# Reference
### paper:
[Arcface: Additive angular margin loss for deep face recognition] (CVPR 2019)

https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html

    @InProceedings{Deng_2019_CVPR,
    author = {Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
    title = {ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
    }
    
### code: 
아래 github 의 아키텍쳐를 참고하여 재현함

https://github.com/ZhaoJ9014/face.evoLVe


# Environments
```
conda env create -f FaceRecog.yml
conda activate python37
```

# Dataset 다운 주소
다음 구글 드라이브 링크에서 부터 각 데이터셋 다운받아 /data 디렉토리에 압축해제  
https://drive.google.com/drive/folders/1mFQk2hzT32d1JRk7bOHgm1mnXoN6QClB?usp=sharing


데이터셋은 총 5개로 구성\
  1.학습 데이터셋 : MS1MV2, CASIA\
  2.Validation 데이터셋 : AgeDB, CFP, LFW


압축 해제 이후 다음과 같은 폴더 형식을 갖는다\

    |── data \
        ├──> agedb_30 \
        ├──> cfp_align_112\
        ├──> imgs_casia \
        ├──> ms1m_align_112 \
        └──> lfw_align_112


# Directory 설명
    |── applications
        └──> align : 데이터셋을 각 face의 랜드마크를 활용해 align하는데 사용되는 코드
    |── checkpoints : Training checkpoint
        └──> {Train_configuration}_{Train_start_date} : 각 학습 과정마다 train configuration과 train시작 시간을 이름으로 갖고 저장되는 checkpoint 폴더 
            ├──> Backbone_{epoch}_{batch}.pth : 해당 학습 과정, epoch, batch의 Backbone checkpoint
            ├──> Head_{epoch}_{batch}.pth :  해당 학습 과정, epoch, batch의 Head checkpoint
            ├──> Optimizer_{epoch}_{batch}.pth :  해당 학습 과정, epoch, batch의 Optimizer checkpoint
            └──> Evaluation.txt : 해당 학습과정의 Validation결과 
    |── checkpoints_best : Best Training checkpoint
        └──> {Train_configuration}_{Train_start_date} : 각 학습 과정마다 train configuration과 train시작 시간을 이름으로 갖고 저장되는 checkpoint 폴더 
            ├──> Backbone_Best.pth : 해당 학습 과정의 best Backbone checkpoint
            ├──> Head_Best.pth :  해당 학습 과정의 best Head checkpoint
            └──> Optimizer_Best.pth : 해당 학습 과정의 best Optimizer checkpoint
    |── data (모두 다 기존 데이터셋의 aligned된 버전)
        ├──> agedb_30 : validation으로 사용되는 AgeDB30 데이터셋
        ├──> cfp_align_112 : validation으로 사용되는 CFP_FP & CFP_FF 데이터셋
        ├──> imgs_casia : train용으로 사용되는 CASIA 데이터셋
        ├──> imgs_train : train용으로 사용되는 CASIA 데이터셋의 작은 버전
        └──> lfw_align_112 : validation으로 사용되는 LFW 데이터셋
    |── model
        ├──> backbone : ArcFace 아키텍쳐에 Backbone으로 사용되는 IR_SE_50에 대한 모듈
        ├──> head : ArcFace 아키텍쳐에 Head으로 사용되는 모듈
        └──> loss : ArcFace 아키텍쳐에 Loss으로 사용되는 Focal Loss에 대한 모듈
    |── log : 학습과정중 log파일
    |── util : 다양한 기타 사용 함수들 폴더
    |── config.py : 입력 argument를 관리하는 파일
    |── FaceRecog.yml : 가상환경 파일
    |── validation.py : validation용 코드(AgeDB, CFP, LFW에대한 validation)
    └── train.py : train용 코드


# 코드 실행 가이드 라인

## === Train ===
학습용 코드 - train.py

### 1) dataset 준비
   위 dataset 다운 주소에서 각 dataset 압축파일을 다운받아 /data/ 폴더에 AgeDB, cfp_align_112, imgs_casia, lfw_align_112폴더 생성\
   https://drive.google.com/drive/folders/1mFQk2hzT32d1JRk7bOHgm1mnXoN6QClB?usp=sharing

### 2) 실행

   아래 명령어를 통해 실행한다
   
   python train.py --gpu_num={[사용할 gpu들의 index]}

      EX. python train.py --gpu_num=[0,1] 
   
   기본 epoch는 50, batch size는 64으로 되어있으며, 변경하고 싶을 시 아래와 같이 추가한다\
   python train.py --gpu_num={[사용할 gpu들의 index]} --epochs={epoch_num} --batch_size={batch_size}
   

   추가적으로 기존 학습 데이터의 Low-resolution버전으로 모델을 학습할 수 있다.\
   기존 학습 데이터는 112x112 크기를 가진다.\
   LR_train를 True로 추가해주고 원하는 LR_scale을 argument로 주면된다

   python train.py --gpu_num={[사용할 gpu들의 index]} --epochs={epoch_num} --batch_size={batch_size} --LR_train={True/False} --LR_scale={LR scale int}

      EX. python train.py --gpu_num=[0,1] --epochs=100 --batch_size=32 --LR_train=True --LR_scale=2 


### 3) 결과 저장
   학습이 시작되면 ./checkpoints/ 폴더 안에 아래와 같이 모든 epoch에 대한 checkpoint가 생성된다

        |── checkpoints
           ├──> {Train_configuration}_{Train_start_date}
              ├──> Backbone_{epoch}_{batch}.pth
              ├──> Head_{epoch}_{batch}.pth
              ├──> Optimizer_{epoch}_{batch}.pth 
              └──> Evaluation.txt 
   이에 더불어, checkpoints중 가장 좋은 validation결과를 갖는 checkpoint는 ./best_checkpoints/ 폴더 안에 아래와 같이 저장된다

        |── checkpoints_best 
           └──> {Train_configuration}_{Train_start_date} 
              ├──> Backbone_Best.pth
              ├──> Head_Best.pth 
              └──> Optimizer_Best.pth   


## === 학습된 ckpt ===

혹은 아래 링크에서 미리 학습한 ckpt 파일을 다운 받아 checkpoints 폴더 안에 배치한 이후
argument로 --resume_dir을 지정해주어 validation을 진행한다

구글 드라이브 주소 : 
https://drive.google.com/file/d/1dm03XeMz2HdpPbCQ-yRePlDTIpW4kDyI/view?usp=sharing

## === Validation ===

학습한 checkpoint를 validation데이터셋(CFP, LFW, AgeDB)에 대해 validation진행한다

### 1) 데이터셋 확인
   ./data 내에 있는 CFP, LFW, AgeDB에 대해 validation을 진행한다

### 2) 코드 실행
   아래 명령어를 통해 실행한다. 
 
   python validation.py --gpu_num={[사용할 gpu들의 index]} --resume_dir={학습한 checkpoint 폴더 디렉토리}
   
      EX. python validation.py --gpu_num=[0,1] --resume_dir = "/data/parkjun210/Final_Detect_Recog/Code_face_recog/checkpoints_best/HRTRAIN_2022-07-28-19-15"
   
   따로 resume_dir을 지정해주지않으면 checkpoints_best안에 있는 제일 최근 checkpoint 폴더로 validation이 진행된다.
   
   추가적으로 LR, SR argument를 줄 수 있다
   Validation dataset의 LR버전에 대한 validation결과를 보고싶다면 --LR_eval=True
   LR버전의 Validation dataset의 Super Resolution한 이미지들에 대한 결과를 보고싶다면 --SR_eval=True

      EX. python validation.py --gpu_num=[0,1] --resume_dir = "/data/parkjun210/Final_Detect_Recog/Code_face_recog/checkpoints_best/HRTRAIN_2022-07-28-19-15" --LR_eval=True --SR_eval=True

### 3) 결과
   결과는 아래와 같이 각 Validation 데이터셋에 대한 결과와
   평균 Accuracy를 출력
