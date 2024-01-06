# Autonomous Exploration Under Uncertainty via Deep Reinforcement Learning on Graphs
This repository contains code for robot exploration under uncertainty that uses graph neural networks (GNNs) in conjunction with deep reinforcement learning (DRL), enabling decision-making over graphs containing exploration information to predict a robot’s optimal sensing action in belief space. A demonstration video can be found [here](https://youtu.be/e7uM03hMZRo).

<p align='center'>
    <img src="/doc/exploration_graph.png" alt="drawing" width="1000" />
</p>

<p align="center">
  <img src="/doc/test_largermap.gif" alt="drawing" width="1000" /> 
</p>

## Dependency
- Python 3
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/#)
  ```shell
  # torch-***+cu*** should be compatible with your pytorch version
  pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
  pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
  pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
  pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.0+cu111.html
  pip install torch-geometric
  ```
- [gtsam](http://www.borg.cc.gatech.edu/sites/edu.borg/files/downloads/gtsam.pdf) (Georgia Tech Smoothing and Mapping library)
  ```shell
  git clone --branch 4.0.3 https://github.com/borglab/gtsam.git
  cd gtsam
  mkdir build && cd build
  cmake ..
  make check
  sudo make install
  ```
- [pybind11](https://github.com/pybind/pybind11) (pybind11 — Seamless operability between C++11 and Python)
  ```shell
  git clone https://github.com/pybind/pybind11.git
  cd pybind11
  mkdir build && cd build
  cmake ..
  sudo make install
  ```

<br>
After completing the installation of the above packages, we recommend run the following command to make sure the dynamic libraries are linked correctly.

```shell
sudo ldconfig
```

## Compile
You can use the following commands to download and compile the package.
```shell
git clone https://github.com/RobustFieldAutonomyLab/DRL_graph_exploration.git
cd DRL_graph_exploration
mkdir build && cd build
cmake ..
make # make -j8
```

<br>

Please use the following command to add the build folder to the python path of the system.
ps: `env_name` is the anaconda env that you create (If you run the code in an anaconda env).

```shell
export PYTHONPATH=/path/to/folder/DRL_graph_exploration/build:$PYTHONPATH
export LIBRARY_PATH=/path/to/folder/anaconda3/envs/env_name/lib/:$LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/folder/anaconda3/envs/env_name/lib:$LD_LIBRARY_PATH
```
## Issues
1. There is an unsolved memory leak issue in the C++ code. So we use the python subprocess module to run the simulation training. The data in the process will be saved and reloaded every 10000 iterations.

2. The code was compiled on a local machine running **Ubuntu 20.04** as the operating system with a GTX 3090 GPU with Cuda 11.2 to speed up training/inference with pytorch. However, because of very large amount of training (1 million epochs as shown in figure 3b in the paper), training was not possible as it took about 2 hours to run for about 12,000 epochs, which is about 83 hours total without checkpointing implemented in the original code. Due to the difficulties compiling on a cloud computing server, we trained for a limited number of epochs (10,000) on the local computer with a GPU.

3. If run in on ubuntu 20.04, please add `set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")` in CMakeLists.txt.

4. If you have a error `AttributeError:module 'distutils' has no attribute 'version'`, please install the appropriate version of distutils.
    ```shell
    pip install setuptools==59.5.0
    ```


## How to Run?
- To run the saved policy:
    ```shell
    cd DRL_graph_exploration/scripts
    python3 test.py
    ```
- To show the average reward during the training:
    ```shell
    pip install tensorboard # install tensorboard first
    cd DRL_graph_exploration/data
    tensorboard --logdir=torch_logs
    ```
- To train your own policy:
    ```shell
    cd DRL_graph_exploration/scripts
    python3 train.py
    ```
 

## Cite

Please cite [our paper](http://ras.papercept.net/images/temp/IROS/files/0778.pdf) if you use any of this code: 
```
@inproceedings{chen2020autonomous,
  title={Autonomous Exploration Under Uncertainty via Deep Reinforcement Learning on Graphs},
  author={Chen, Fanfei and Martin, John D. and Huang, Yewei and Wang, Jinkun and Englot, Brendan},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={6140--6147},
  year={2020},
  organization={IEEE}
}
```

## Reference
- [em_exploration](https://github.com/RobustFieldAutonomyLab/em_exploration)
