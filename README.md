## Train

To specify the training method & dataset, please modify the configurations in dropen\_train.py by adding annotations to certain lines of code. For example, you can change the training method to Naive\_adv\_trainer, Naive\_normal\_trainer or TRS_trainer. You can also switch between models to get two types models that classify MNIST or CIFAR10 data.

    python3 dropen_train.py 

## Resume training for a certain model

Please refer to the args predefined in dropen\_resume.py. The need to resume training a model typically emerges in the situation of CIFAR10 models.

    python3 dropen_resume.py 


## Test & Attack models
To switch between datasets, please alse modify the model structure in dropen\_attack.py
### Test for CIFAR10
    python3 dropen_attack.py --dataset CIFAR10 --base_classifier ./dropen_model/TRS.pth.tar --attack_type pgd --adv-eps 0.02

### Test for MNIST
	python3 dropen_attack.py --dataset MNIST --base_classifier ./dropen_model/TRS.pth.tar --attack_type pgd --adv-eps 0.02

## Trained Models

Trained models for adversarial attack validation can be found in https://cloud.tsinghua.edu.cn/d/1fe4412998404ca3ac03/
