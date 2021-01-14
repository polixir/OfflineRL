import argparse
import os
import json
import oss2

def upload_result(task_name : str, algo_name : str, result : int, ip : str):
    file_name = task_name + ',' + algo_name + '.txt'
    with open(os.path.join('/tmp', file_name), 'w') as f:
        f.write(str(result))
        f.write('\n')
        f.write(ip)
    access_key_id  = "LTAIGoWWsroWIQAo"
    access_key_secret = "tYcybZDpgA48DSXrGpCA6kxEcbIrZM"
    bucket_name = "polixir-ai"
    endpoint = "https://oss-cn-shanghai.aliyuncs.com"
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    bucket.put_object_from_file(os.path.join("eval_res", file_name), os.path.join('/tmp', file_name)) 

def get_host_ip():
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

def find_result(exp_name : str):
    lib_path = os.path.abspath(os.path.dirname(__file__))
    aim_path = os.path.join(lib_path, 'batchrl_tmp', '.aim')
    exp_path = os.path.join(aim_path, exp_name)
    max_result = - float('inf')
    for name in os.listdir(exp_path):
        if len(name) > 6: # not index
            data_file = os.path.join(exp_path, name, 'objects', 'map', 'dictionary.log')
            with open(data_file, 'r') as f:
                data = json.load(f)
            result = data['__METRICS__']['Reward_Mean_Env'][0]['values']['last']
            max_result = max(result, max_result)
    return max_result

def upload_json(exp_name : str):
    target_ip = '10.200.0.41'
    user_name = 'ubuntu'
    passwd = 'ubuntu'
    template = 'sshpass -p {passwd} scp -o StrictHostKeyChecking=no {source_file} {user_name}@{target_ip}:~/walker_eval/{exp_name}/{target_file}'

    # create folder in remote 
    os.system(f'sshpass -p {passwd} ssh -o StrictHostKeyChecking=no {user_name}@{target_ip} "cd walker_eval && mkdir {exp_name}"')

    lib_path = os.path.abspath(os.path.dirname(__file__))
    aim_path = os.path.join(lib_path, 'batchrl_tmp', '.aim')
    exp_path = os.path.join(aim_path, exp_name)    

    index = 0
    for name in os.listdir(exp_path):
        if len(name) > 6: # not index
            source_file = os.path.join(exp_path, name, 'metric_logs.json')
            target_file = f'metric_logs_{index}.json'
            index += 1
            os.system(template.format(passwd=passwd, source_file=source_file, user_name=user_name, target_ip=target_ip, exp_name=exp_name, target_file=target_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='full name of the task, like HalfCheetah-v3-low-99-train-noise')
    parser.add_argument('--algo', type=str, help='select from `bc` and `mail`')
    
    args = parser.parse_args()

    task = args.task
    algo = args.algo

    split_task = task.split('-')

    if split_task[1] == 'v3':
        task_name = '-'.join(split_task[:2])
        split_task = split_task[2:]
    else:
        task_name = split_task[0]
        split_task = split_task[1:]

    level = split_task[0]
    train_num = int(split_task[1])
    others = split_task[3:]
    val_num = (train_num + 1) // 10
    algo = args.algo
    exp_name = '-'.join([task_name, level, str(train_num), *others, algo, 'eval'])

    training_command = f'python BatchRL/examples/train_tune.py --algo_name {algo} --exp_name {exp_name} --task {task}'
    
    print(f'running command: {training_command}')
    os.system(training_command)

    ip = get_host_ip()
    r = int(find_result(exp_name))
    upload_result('-'.join([task_name, level, str(train_num)]), algo, r, ip)
    upload_json(exp_name)