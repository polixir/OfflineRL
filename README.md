# BatchRl 
Batchrl is a repository for Batch RL(batch reinforce learning or offline reinforce learning).

## Install bactchrl

```
pip install -r requirments.txt

pip install -e .
```

## Example

```
python examples/test_revive.py --algo_name=cql --exp_name=halfcheetah-medium-v0 --task HalfCheetah-v3-low-99-train

python examples/train_tune.py --algo_name=cql --exp_name=halfcheetah-medium-v0 --task HalfCheetah-v3-low-99-train --aim_path /tmp/.aim/
```