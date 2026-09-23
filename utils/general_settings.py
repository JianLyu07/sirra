import torch,logging,datetime,json,re,os,warnings
from basic_function import get_abs_path

# 配置日志
def setup_logging():
    start_time = datetime.datetime.now()
    file_name = f'../../log/{start_time.strftime("%Y_%m_%d_%H_%M_%S")}.log'     
    file_name = get_abs_path(file_name)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    # logging.basicConfig(level=logging.INFO,format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    #                 datefmt='## %Y-%m-%d %H:%M:%S')
    
    file_handler = logging.FileHandler(file_name,encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    # file_handler.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s'))
    
    adapter_logger = logging.getLogger("peft.tuners.tuners_utils")
    adapter_logger.setLevel(logging.WARNING)
       
    logger = logging.getLogger()
    if not hasattr(logger, 'file_handler_added'):
        logger.addHandler(file_handler)
        logger.file_handler_added = True

# 获取系统的gpu数量        
def generate_device_list():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
    else:
        device_count = 0
        
    if device_count == 0:
        return ""  
    devices = list(range(device_count))   
    return ",".join(map(str, devices))

class ConfigManager:
    '''
    管理不同模型训练配置的类
    '''
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = {}
        with open(self.config_path, 'r',encoding='utf-8') as f:
            raw_json_text = f.read()
        json_text_without_comments = self.remove_comments(raw_json_text)
        try:
            self.config = json.loads(json_text_without_comments)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")

    def remove_comments(self, text):
        # 移除C风格注释
        text = re.sub(r'//.*', '', text)
        # 移除Python风格注释
        text = re.sub(r'#.*', '', text)
        # 移除块注释
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text
    
    def get_full_config(self):
        return self.config
