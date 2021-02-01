# offlinerl 
offlinerl is a repository for Batch RL(batch reinforce learning or offline reinforce learning).

## Install bactchrl

```
pip install -e .
```

## Example

```
python examples/train_task.py --algo_name=cql --task HalfCheetah-v3 --task_data_type low --task_train_num 99

python examples/train_tune.py --algo_name=cql --exp_name=halfcheetah-medium-v0 --task HalfCheetah-v3-low-99-train --aim_path /tmp/.aim/
```