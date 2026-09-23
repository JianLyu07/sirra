import sys,os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_environment():  
    py_dir = os.path.dirname(os.path.realpath(__file__)) 
    if py_dir not in sys.path:
        sys.path.append(py_dir)
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"  
    from general_settings import generate_device_list  
    os.environ['CUDA_VISIBLE_DEVICES'] = generate_device_list()  
    
setup_environment()