# Imitation-Learning-using-Attention-Mechanism

[Python virtual environment setting]
    
    $ conda create -n code_test python=3.8.10
    $ conda activate code_test
    $ pip install numpy
    $ conda install pytorch pytorch-cuda pytorch –c nvidia

[Training phase using IL dataset]
    
    $ python carla_ad.py
    pygame 윈도우에서 숫자 2 입력 후 직접 주행으로 데이터 수집
    $ python train.py

[Test for Real-Time IL Autonomous driving]
(In case that CARLA server is working)

    $ python carla_ad.py
    pygame 윈도우에서 숫자 3 입력
