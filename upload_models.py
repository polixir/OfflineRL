import os
import json

if __name__ == '__main__':
    copy_command_template = 'sshpass -p ubuntu scp -r -o StrictHostKeyChecking=no {source_file} ubuntu@10.200.0.41:~/offline_models/{target_file}'
    mkdir_command_template = 'sshpass -p ubuntu ssh -o StrictHostKeyChecking=no ubuntu@10.200.0.41 mkdir ~/offline_models/{target_file}'

    aim_folder = 'batchrl_tmp/.aim'
    for logname in os.listdir(aim_folder):
        if not 'final' in logname: continue
        log_folder = os.path.join(aim_folder, logname)
        os.system(mkdir_command_template.format(target_file='/'.join([logname])))
        for exp_id in os.listdir(log_folder):
            if exp_id == 'index': continue
            exp_folder = os.path.join(log_folder, exp_id)
            os.system(mkdir_command_template.format(target_file='/'.join([logname, exp_id])))

            # load json
            with open(os.path.join(exp_folder, 'metric_logs.json'), 'r') as f:
                metrics = json.load(f)

            # find groundtruth
            gt = {}
            for k, v in metrics.items():
                if not "Reward_Mean_Env" in v.keys(): continue
                gt[k + '.pt'] = v["Reward_Mean_Env"]

            # save gt
            gt_file = '/tmp/gt.json'
            with open(gt_file, 'w') as f:
                json.dump(gt, f, indent=2)
            os.system(copy_command_template.format(source_file=gt_file, target_file='/'.join([logname, exp_id, 'gt.json'])))

            # copy model
            os.system(mkdir_command_template.format(target_file='/'.join([logname, exp_id, 'models'])))
            for model_file in gt.keys():
                os.system(copy_command_template.format(source_file=os.path.join(exp_folder, 'models', model_file), target_file='/'.join([logname, exp_id, 'models', model_file])))