import os
import json
from tqdm import tqdm

def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

if __name__ == '__main__':
    copy_command_template = 'cp {source_file} {target_file}'

    aim_folder = 'batchrl_tmp/.aim'
    tmp_folder = '/tmp/offline_models'
    make_dir(tmp_folder)

    for logname in tqdm(filter(lambda x: 'final' in x, os.listdir(aim_folder))):
        log_folder = os.path.join(aim_folder, logname)
        for exp_id in os.listdir(log_folder):
            if exp_id == 'index': continue
            exp_folder = os.path.join(log_folder, exp_id)

            try:
                # load json
                with open(os.path.join(exp_folder, 'metric_logs.json'), 'r') as f:
                    metrics = json.load(f)

                # find groundtruth
                gt = {}
                for k, v in metrics.items():
                    if not "Reward_Mean_Env" in v.keys(): continue
                    gt[k + '.pt'] = v["Reward_Mean_Env"]

                make_dir(os.path.join(tmp_folder, logname, exp_id))

                # save gt
                with open(os.path.join(tmp_folder, logname, exp_id, 'gt.json'), 'w') as f:
                    json.dump(gt, f, indent=2)

                # copy model
                make_dir(os.path.join(tmp_folder, logname, exp_id, 'models'))
                for model_file in gt.keys():
                    os.system(copy_command_template.format(
                        source_file=os.path.join(exp_folder, 'models', model_file), 
                        target_file=os.path.join(tmp_folder, logname, exp_id, 'models', model_file)))
            except Exception as e:
                print(e)

    os.system('sshpass -p ubuntu scp -r -o StrictHostKeyChecking=no /tmp/offline_models/* ubuntu@10.200.0.41:~/offline_models/')