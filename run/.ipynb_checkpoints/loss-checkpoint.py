import os
import sys
import subprocess

device_id = 0
CoNLL2000_dir = "../data/CoNLL2000"
ckpt_file = "../ckpt_lstm_crf/trained.ckpt"

os.makedirs('ms_log', exist_ok=True)
cur_dir = os.getcwd()
os.environ['GLOG_log_dir'] = os.path.join(cur_dir, 'ms_log')
os.environ['GLOG_logtostderr'] = '0'

base_path = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(base_path, '..', 'default_config.yaml')

command = f"python ../accLoss.py --config_path={config_file} --device_target=Ascend --device_id={device_id} --data_CoNLL_path={CoNLL2000_dir} --ckpt_path={ckpt_file} --build_data=False --preprocess=true --preprocess_path=./preprocess"

with open("result.txt", "w") as log_file:
    subprocess.run(command, shell=True, stderr=subprocess.STDOUT, stdout=log_file)
