# vgn

代码主要来自 https://github.com/ethz-asl/vgn/tree/devel , 即论文 [Volumetric Grasping Network: Real-time 6 DOF Grasp Detection in Clutter](http://arxiv.org/abs/2101.01132). `src/inference` 来自原存储库中的 `src/vgn`, `src/robot_helpers` 来自[这个](https://github.com/mbreyer/robot_helpers)仓库.

## 安装依赖

    conda create -c conda-forge -n vgn python=3.10 pytorch numpy scipy scikit-image catkin_pkg

## 训练模型

使用原仓库并遵循其说明以进行训练，依赖安装方式为：

    conda create -c conda-forge -n vgn python numpy=1.19 mpi4py pandas pybullet pytorch pytorch-ignite pyyaml scikit-image scipy tensorboard tqdm mayavi
    pip install open3d

使用numpy=1.19以兼容原仓库代码中的`np.long`

需要修改源代码simulation.py中的文件名

生成数据（一堆方块）

    ln -s /home/yangzhuo/references/robot_helpers/robot_helpers /home/yangzhuo/references/vgn/scripts/robot_helpers
    ln -s /home/yangzhuo/references/vgn/src/vgn /home/yangzhuo/references/vgn/scripts/vgn
    LD_LIBRARY_PATH=/home/yangzhuo/mambaforge/envs/vgn/lib  mpirun -np 6 python3 scripts/generate_data.py --root=data/grasps/blocks

生成pakced（需要修改源代码generate_data.py中generate_pile为generate_packed）

    LD_LIBRARY_PATH=/home/yangzhuo/mambaforge/envs/vgn/lib  mpirun -np 6 python3 scripts/generate_data.py --root=data/grasps/pile --cfg cfg/sim/pile.yaml --count 750000 --mode 1 && LD_LIBRARY_PATH=/home/yangzhuo/mambaforge/envs/vgn/lib  mpirun -np 6 python3 scripts/generate_data.py --root=data/grasps/packed --cfg cfg/sim/packed.yaml --count 2000000 --mode 0

packed.yaml
```
sim:
  urdf_root: assets/urdfs
  gui: False
  lateral_friction: 1.0

object_urdfs: assets/urdfs/packed/train
object_count_lambda: 4
scaling:
  low: 0.8
  high: 1.5
scene: packed
max_view_count: 5
scene_grasp_count: 200

metric: dynamic_with_approach
```

使用process_data.ipynb

    ln -s /home/yangzhuo/references/vgn/data/grasps/packed/*.npz /home/yangzhuo/references/vgn/data/grasps/train/ -r
    ln -s /home/yangzhuo/references/vgn/data/grasps/pile/*.npz /home/yangzhuo/references/vgn/data/grasps/train/ -r

生成训练数据集

    python scripts/create_dataset.py  data/grasps/train data/dataset/train

训练与可视化

    python scripts/train_vgn.py --dataset data/dataset/train --augment --logdir logs
    tensorboard --logdir logs
