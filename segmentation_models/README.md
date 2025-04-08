### 구성 환경
| 구성 요소       | 내용                                       |
|-------------|------------------------------------------|
| 백본(Encoder) | EfficientNetB0 (ImageNet pretrained)     |
| 모델          | U-Net (segmentation_models)              |
| 입력 크기       | (512, 512, 3)                            |
| 출력 채널       | 1(binary segmentation)                   |
| 손실함수        | Dice + BCE loss 조합                       |
| 메트릭         | IoU, Dice, Accuracy                      |
| 데이터 증강      | albumentations                           |
| 콜백          | EarlyStopping, ReduceLR, ModelCheckpoint |

