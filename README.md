# OfflineRL 

OfflineRL is a repository for Offline RL (batch reinforcement learning or offline reinforcement learning).

## Re-implemented Algorithms
### Model-free methods
- **CRR**: Wang, Ziyu, et al. “Critic Regularized Regression.” Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 7768–7778. [paper](https://arxiv.org/abs/2006.15134)
- **CQL**: Kumar, Aviral, et al. “Conservative Q-Learning for Offline Reinforcement Learning.” Advances in Neural Information Processing Systems, vol. 33, 2020. [paper](https://arxiv.org/abs/2006.04779) [code](https://github.com/aviralkumar2907/CQL)
- **PLAS**: Zhou, Wenxuan, et al. “PLAS: Latent Action Space for Offline Reinforcement Learning.” ArXiv Preprint ArXiv:2011.07213, 2020.
 [website](https://sites.google.com/view/latent-policy) [paper](https://arxiv.org/abs/2011.07213) [code](https://github.com/Wenxuan-Zhou/PLAS)
- **BCQ**: Fujimoto, Scott, et al. “Off-Policy Deep Reinforcement Learning without Exploration.” International Conference on Machine Learning, 2018, pp. 2052–2062. [paper](https://arxiv.org/abs/1812.02900) [code](https://github.com/sfujim/BCQ)
- **PRDC**: Ran, Yuhang, et al. “Policy Regularization with Dataset Constraint for Offline Reinforcement Learning.” International Conference on Machine Learning, 2023, pp. 28701-28717. [paper](https://arxiv.org/abs/2306.06569) [code](https://github.com/LAMDA-RL/PRDC)
### Model-based methods
- **BREMEN**: Matsushima, Tatsuya, et al. “Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization.” International Conference on Learning Representations, 2021. [paper](https://openreview.net/forum?id=3hGNqpI4WS) [code](https://github.com/matsuolab/BREMEN)
- **COMBO**: Yu, Tianhe, et al. "COMBO: Conservative Offline Model-Based Policy Optimization." arXiv preprint arXiv:2102.08363 (2021). [paper](https://arxiv.org/abs/2102.08363)
- **MOPO**: Yu, Tianhe, et al. “MOPO: Model-Based Offline Policy Optimization.” Advances in Neural Information Processing Systems, vol. 33, 2020. [paper](https://papers.nips.cc/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html) [code](https://github.com/tianheyu927/mopo)
- **MAPLE**: Xiong-Hui Chen, et al. "MAPLE: Offline Model-based Adaptable Policy Learning". Advances in Neural Information Processing Systems, vol. 34, 2021. [paper](https://proceedings.neurips.cc/paper/2021/hash/470e7a4f017a5476afb7eeb3f8b96f9b-Abstract.html) [code](https://github.com/xionghuichen/MAPLE)
- **MOBILE**: Yihao Sun, et al. "Model-Bellman Inconsistency for Model-based Offline Reinforcement Learning". Proceedings of the 40th International Conference on Machine Learning, PMLR 202:33177-33194, 2023. [paper](https://proceedings.mlr.press/v202/sun23q.html) [code](https://github.com/yihaosun1124/mobile)

## Install Datasets
### NeoRL

```shell
git clone https://agit.ai/Polixir/neorl.git
cd neorl
pip install -e .
```

For more details on use, please see [neorl](https://agit.ai/Polixir/neorl).

### D4RL (Optional)
```shell
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

For more details on use, please see [d4rl](https://github.com/rail-berkeley/d4rl).

## Install offlinerl

```shell
pip install -e .
```

## Example

```python
# Training in HalfCheetah-v3-L-9 task using default parameters of cql algorithm
python examples/train_task.py --algo_name=cql --exp_name=halfcheetah --task HalfCheetah-v3 --task_data_type low --task_train_num 100

# Parameter search in the default parameter space using the cql algorithm in the HalfCheetah-v3-L-9 task
python examples/train_tune.py --algo_name=cql --exp_name=halfcheetah --task HalfCheetah-v3 --task_data_type low --task_train_num 100

# Training in D4RL halfcheetah-medium task using default parameters of cql algorithm (D4RL need to be installed)
python examples/train_d4rl.py --algo_name=cql --exp_name=d4rl-halfcheetah-medium-cql --task d4rl-halfcheetah-medium-v0
```

**Parameters:**

- ​**algo_name**:  Algorithm name . There are now bc, cql, plas,  bcq and mopo algorithms available.
- ​**exp_name**:  Experiment name for easy visualization using aim.
- ​**task**: Task name, See [neorl](https://agit.ai/Polixir/neorl/wiki/Tasks) for details.
- ​**task_data_type**: Data level. Each task collects data using low, medium, and high level strategies in [neorl](https://agit.ai/Polixir/neorl).
- ​**task_train_num**:  Number of training data trajectories. For each task, neorl provides training data for up to 10000 trajectories.



## View experimental results
We use **Aim** to store and visualize results. Aim is an experiment logger that is easy to manage thousands of experiments. For more details, see [aim](https://github.com/aimhubio/aim). 

To visualize results in this repository:
```shell
cd offlinerl_tmp
aim up
```
Then you can see the results on http://127.0.0.1:43800.