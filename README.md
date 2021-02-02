# OfflineRL 

OfflineRL is a repository for Offline RL(batch reinforce learning or offline reinforce learning).

## Install newrl

```shell
git clone https://agit.ai/Polixir/newrl.git
cd newrl
pip install -e .
```

For more details on use, please see [newrl](https://agit.ai/Polixir/newrl)。

## Install bactchrl

```shell
pip install -e .
```

## Example

```python
# Training in HalfCheetah-v3-L-9 task using default parameters of cql algorithm
python examples/train_task.py --algo_name=cql --exp_name=halfcheetah --task HalfCheetah-v3 --task_data_type low --task_train_num 99

# Parameter search in the default parameter space using the cql algorithm in the HalfCheetah-v3-L-9 task
python examples/train_tune.py --algo_name=cql --exp_name=halfcheetah --task HalfCheetah-v3 --task_data_type low --task_train_num 99
```

**Parameters:**

- ​            **algo_name**  :  Algorithm name . There are now bc, cql, plas,  bcq and mopo algorithms available.
- ​            **exp_name** :  Experiment name for easy visualization using aim.
- ​            **task**  : Task name, See [newrl](https://agit.ai/Polixir/newrl/wiki/Tasks) for details.
- ​            **task_data_type** : Data level. Each task collects data using a low, medium, and high level strategy in [newrl](https://agit.ai/Polixir/newrl).
- ​            **task_train_num** :  Number of train data trajectories. For each task, newrl provides training data for up to 9999 trajectories.



## View experimental results

```shell
cd offlinerl_tmp

aim up
```

For more details on use, see [aim](https://github.com/aimhubio/aim)。

