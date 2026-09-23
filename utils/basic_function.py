import json,os,pickle,torch,warnings,logging,random
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import islice

py_directory = os.path.dirname(os.path.realpath(__file__))

def get_key_position(dict, key_to_find):
    # 已知字典中某个元素的键，查找这个元素在字典中的序号
    try:
        # 遍历字典的键，并计数直到找到目标键
        for index, key in enumerate(dict):
            if key == key_to_find:
                return index  # 返回基于0的索引位置
        return -1  # 如果没有找到键，则返回-1
    except TypeError:
        # 如果d不是可迭代的或者key_to_find不是正确的类型，抛出异常
        raise ValueError("提供的参数不正确")

def get_abs_path(relative_path):
    """
    通过相对路径获取绝对路径
    """
    abs_path = os.path.abspath(os.path.join(py_directory, relative_path))
    return abs_path

def load_json(relative_json_path):
    """
    通过相对路径加载json文件
    """
    abs_json_path = get_abs_path(relative_json_path)
    with open(abs_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(relative_json_path,param_to_save):
    """
    通过相对路径把一个变量的内容保存到json文件
    """
    abs_json_path = get_abs_path(relative_json_path)
    with open(abs_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(param_to_save, json_file, ensure_ascii=False, indent=4)
        print("json文件保存成功")

def load_pickle(relative_pickle_path):
    """
    通过相对路径加载pickle文件
    """
    abs_pickle_path = get_abs_path(relative_pickle_path)
    with open(abs_pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(relative_pickle_path,param_to_save):
    """
    通过相对路径把一个变量的内容保存到pickle文件
    """
    abs_pickle_path = get_abs_path(relative_pickle_path)
    with open(abs_pickle_path, 'wb') as f:
        try:
            pickle.dump(param_to_save, f)
            print("pickle文件保存成功")
        except:
            print("pickle文件保存失败")
            
def chat_qwen(model, tokenizer, prompt, max_new_tokens, gnmt_alpha=0.2):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")
        with torch.no_grad():
            # 设置 output_scores=True 来获取 logits
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,  # 获取输出 logits
                return_dict_in_generate=True  # 返回字典格式的输出
            )
    
    # 提取生成的 token IDs
    generated_ids = outputs.sequences
    # 提取输入部分的长度
    input_length = model_inputs.input_ids.shape[1]
    # 只保留生成的响应部分（去掉输入）
    response_ids = generated_ids[:, input_length:]
    
    # 解码生成的响应
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    
    # 收集所有生成步骤的 logits
    # outputs.scores 是一个包含每个生成步骤 logits 的元组
    logits = torch.stack(outputs.scores, dim=1)  # 形状: [batch_size, sequence_length, vocab_size]
    
    # 因为我们只有一个样本，所以取第一个批次
    response_logits = logits[0]  # 形状: [response_length, vocab_size]
    
    # *** 新增：根据生成出的 token 计算 log 概率
    # response_ids: [1, response_length]
    token_ids = response_ids[0]  # 形状: [response_length]

    # 对每一步的 logits 做 log_softmax，然后取对应 token 的 logP
    log_probs = []
    for step_idx in range(response_logits.size(0)):
        step_logits = response_logits[step_idx]          # [vocab_size]
        step_log_probs = F.log_softmax(step_logits, dim=-1)  # [vocab_size]
        token_id = token_ids[step_idx]
        log_prob_token = step_log_probs[token_id]
        log_probs.append(log_prob_token)

    if len(log_probs) > 0:
        log_probs_tensor = torch.stack(log_probs)        # [response_length]

        # ==========================================================
        # 1. Mean Token Log-Probability（原论文 Eq.14）
        #
        # conf = (1/L) * sum log P(y_t)
        #
        # 即生成序列所有 token log probability 的平均值。
        # 越接近 0，表示模型对生成 token 越自信。
        # ==========================================================
        avg_log_prob = log_probs_tensor.mean().item()


        # ==========================================================
        # 2. Raw Sequence Log-Probability
        #
        # score_raw = sum log P(y_t)
        #
        # 不进行长度归一化，直接计算整个生成序列的
        # autoregressive sequence log-likelihood。
        # ==========================================================
        raw_log_prob = log_probs_tensor.sum().item()


        # ==========================================================
        # 3. GNMT-style Length Penalty Score
        #
        # score_gnmt =
        #     sum log P(y_t) / ((5+L)/6)^alpha
        #
        # 采用 GNMT (Wu et al., 2016) 的长度惩罚形式。
        # alpha 默认取 0.2。
        #
        # ==========================================================
        response_length = log_probs_tensor.size(0)

        length_penalty = ((5 + response_length) / 6) ** gnmt_alpha

        gnmt_score = raw_log_prob / length_penalty

    else:
        avg_log_prob = float("-inf")
        raw_log_prob = float("-inf")
        gnmt_score = float("-inf")


    return response, response_logits, float(avg_log_prob), float(raw_log_prob), float(gnmt_score)