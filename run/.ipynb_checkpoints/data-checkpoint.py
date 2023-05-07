import os

os.makedirs("ms_log", exist_ok=True)
CUR_DIR = os.getcwd()
os.environ["GLOG_log_dir"] = os.path.join(CUR_DIR, "ms_log")
os.environ["GLOG_logtostderr"] = "0"


CoNLL2000_DIR = "../data/CoNLL2000"
GLOVE_DIR = "../data/glove"


CUR_DIR = os.getcwd()
os.environ["GLOG_log_dir"] = os.path.join(CUR_DIR, "ms_log")
os.environ["GLOG_logtostderr"] = "0"

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_PATH, "../default_config.yaml")

os.system(f"python ../train.py --config_path={CONFIG_FILE} \
           --data_CoNLL_path={CoNLL2000_DIR} \
           --glove_path={GLOVE_DIR} \
           --device_target=CPU \
           --build_data=True \
           --preprocess=True \
           --preprocess_path=./preprocess > log_build_data.txt 2>&1 &")
