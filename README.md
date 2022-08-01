# SNU: Face detection & recognition

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
**Face detection**

아래 github 의 아키텍쳐를 참고하여 multi-stage로 재현함

https://github.com/biubug6/Pytorch_Retinaface

**Face recognition**

아래 Github code 참조

https://github.com/ZhaoJ9014/face.evoLVe

# 실행 결과 예시 



# Environments
```
conda create -n ENV_NAME python=3.7


conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install opencv-python

pip install tqdm
```


# Directory 설명
    |── Code_face_recog     : Face image recognition code 폴더
    |── data                : image & video inference용 input data가 저장되는 폴더
    |── library             : face recognition에 필요한 Library data(aligned face)를 모아놓는 폴더
    |── pipeline_tools
        ├──> recognition.py : detected face image와 library face image를 비교하는 코드
    |── results             : image & video inference 실행 결과가 저장되는 폴더 
    |── SNU_FaceDetection   : Face detection code 폴더
    |── utils               : 다양한 기타 사용 함수들 폴더
    |── config.py           : 입력 argument를 관리하는 파일
    |── img_inference.py    : image inference 코드
    |── make_library.py     : image에서 Alined face image (112 x 112)를 추출하는 코드
    |── retinaface.yml      : 가상환경 파일
    └── video_inference.py  : video inference 코드




# 코드 실행 가이드 라인


## === 학습된 ckpt ===


**Face recognition**

아래 링크에서 미리 학습한 ckpt 폴더(arcface_casia)를 다운 받아 "Code_face_recog/checkpoints_best" 폴더 안에 배치한다.

Ex. "Code_face_recog/checkpoints_best/arcface_casia/Backbone_Best.pth"

구글 드라이브 주소 : https://drive.google.com/drive/folders/1zjp58d4T3vB6UA6ReEgzdemOoyxTZPbP?usp=sharing

**Face detection**

아래 링크에서 미리 학습한 ckpt 폴더(resnet_anc2_casT_fpn3)를 다운 받아 "Snu_FaceDetection/experiments" 폴더 생성 후 그 안에 배치한다.

구글 드라이브 주소 : https://drive.google.com/drive/folders/1bbxIfmmlhs33uBkTasL6ksnPfabFFpNI?usp=sharing


## === make_library.py ===

--inference_dir 안의 이미지에서 face detection & alignment를 수행한 뒤 --inference_save_folder 에 alinged face images (112 x 112)를 저장한다.

    python make_library.py --gpu_num=0 --inference_dir='sample_library' --inference_save_folder='results_sample_lib'


## === library (face database) ===

make_library.py 로 생성한 aligned face image를 이용하여 face database가 들어있는 폴더

Ex. --library_dir = 'database'

    |── library
        ├──> database 
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

1. 학습된 face detection ckpt, face recognition ckpt 필요
2. library 폴더 안에 face database 폴더 필요
3. data 폴더 안에 inference할 image가 담긴 폴더 필요

Image inference 코드 - img_inference.py (GT 존재해서 AP 측정 가능할 때)

### 1) 데이터셋 확인

  ./library/{--library_dir}/ 폴더 안에 library face database를 넣는다.

  ./data/{--inference_dir}/ 폴더 안에 inference용 이미지를 넣는다.

### 2) 코드 실행

  아래 명령어를 통해 실행한다. 
 
  python img_inference.py 
  
  --gpu_num                 = {사용할 gpu index, int}

  --inference_dir           = {inference용 이미지가 저장된 폴더, default='sample_imgs'}
     
  --experiment_name         = {inference에 사용할 face detection ckpt 폴더가 저장된 폴더, default='resnet_anc2_casT_fpn3'} 
   
  --recog_experiment_name   = {inference에 사용할 face recognition ckpt 폴더가 저장된 폴더, default='casia_HR_LR4'}

  --save_img                = {inference 결과 이미지를 저장할 지 여부, defalut=True}

  --inference_save_folder   = {결과 이미지를 저장할 폴더 이름, default='inference_results'}

  --infer_imsize_same       = {inference용 이미지들의 크기가 일정한지 여부, default=True}

  --library_dir             = {inference에 사용할 library face database가 담긴 폴더 이름, default='class'}
   
   
    python img_inference.py --gpu_num=0 --inference_dir='sample_imgs' --experiment_name='resnet_anc2_casT_fpn3' --recog_experiment_name='casia_HR_LR4' --library_dir='class'
    
### 3) 결과 저장


   tqdm을 통해 진행 상황을 출력하며, 테스트가 종료되면 테스트에 걸린 시간과 AP 결과를 ./results/{--inference_save_folder}/{--inference_dir}_{--inference_save_folder}.txt에 저장한다

        |── inference_results
           ├──> result_images                           : --save_img=True를 줬을 시 inference 이미지를 저장
           └──> inference_dir_inference_results.txt     : image 이름과 그 bbox, face detection 신뢰도 결과값을 결과로 저장

## === Video Inference ===

Image inference와 동일하며 data 폴더 안에 inference할 video가 담긴 폴더를 --inference_dir로 설정
