# Imitation-Learning-using-Attention-Mechanism

[python vertual setting]
conda create -n code test python=3.8.10
$ conda create -n code_test python=3.8.10
$ conda activate code_test
$ pip install numpy
$ conda install pytorch pytorch-cuda pytorch –c nvidia

[Training phase using IL dataset]
$ python carla_ad.py
press '2' key and collect data by manual driving
$ python train.py

[Test for Real-Time IL Autonomous driving]
(In case that CARLA server is working)
$ python carla_ad.py
press '3' on pygame window
