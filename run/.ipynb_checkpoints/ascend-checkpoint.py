import os
import subprocess

DEVICE_ID = 0
CoNLL2000_DIR = "../data/CoNLL2000"
GLOVE_DIR = "../data/glove"
CUR_DIR = os.getcwd()

os.environ['GLOG_log_dir'] = os.path.join(CUR_DIR, 'ms_log')
os.environ['GLOG_logtostderr'] = '0'

BASE_PATH = os.path.abspath(os.path.dirname(__file__))
CONFIG_FILE = os.path.join(BASE_PATH, '..', 'default_config.yaml')

train_cmd = f"python {os.path.join(BASE_PATH, '..', 'startrain.py')} " \
            f"--config_path={CONFIG_FILE} " \
            f"--device_target=Ascend " \
            f"--device_id={DEVICE_ID} " \
            f"--data_CoNLL_path={CoNLL2000_DIR} " \
            f"--glove_path={GLOVE_DIR} " \
            f"--build_data=False " \
            f"--preprocess=true " \
            f"--preprocess_path=./preprocess"

subprocess.Popen(train_cmd, shell=True).communicate()
