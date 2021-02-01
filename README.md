# OfflineRL 
OfflineRL is a repository for Offline RL(batch reinforce learning or offline reinforce learning).

## Install bactchrl

```
pip install -e .
```

## Example

```
python examples/train_task.py --algo_name=cql --task HalfCheetah-v3 --task_data_type low --task_train_num 99

python examples/train_tune.py --algo_name=cql --exp_name=halfcheetah --task HalfCheetah-v3 --task_data_type low --task_train_num 99
```

## View experimental results
```
cd offlinerl_tmp

aim up
```
For more details on use, see [aim](https://github.com/aimhubio/aim)。
