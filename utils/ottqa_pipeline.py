"""
ottqa问题推理的全部流程
"""
import init_utils

import time,logging,torch,random,faiss,sys
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModel,AutoModelForSequenceClassification
from peft import PeftModel
from typing import List

from general_settings import setup_logging
from constant import MY_PROMPT
from basic_function import load_json,load_pickle,get_abs_path,chat_qwen,save_json
from evaluate_script import compute_exact,compute_f1
from model_proc import get_mean_pooling_vec,last_token_pool

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

class OttqaSolver:
    """入参和settings类"""
    def __init__(self,
        rand_seed_value = 42, # 随机数种子

        # 要评估的子集
        eval_dataset = "dev", # "dev" or "test"
        
        # question路径
        devset_path = "../../ottqa_data/released_data/dev.traced.json", # dev集路径
        testset_path = "../../ottqa_data/released_data/test.blind.json", # test集路径
        
        # tables路径
        tables_path = "../../ottqa_data/data/all_plain_tables.json", # 全部table
        table_faiss_path = "../../ottqa_data/gte_multilingual_base_tables.faissindex", # 全部table的index 
        
        # passages路径
        passages_path = "../../ottqa_data/data/all_passages.json", # 全部passage 
        passage_faiss_path = "../../ottqa_data/gte_multilingual_base_passages.faissindex", # 全部passage的index
        
        # early fusion结果的路径（线下提前完成）
        early_fusion_path = "../../ottqa_data/dev_test_possible_tables.pickle",
        
        # question to table retriever
        retriever_worker_path = "../../model/gte-multilingual-base", # retriever预训练模型路径
        retriever_pooling = "first",  # retriever模型的pooling,取值"mean","first","last"
        retriever_q_t_lora_path = "../../model/lora_question_to_table_retreiver", # question to table的lora路径

        # question to table reranker
        ret_reranker_worker_path = "../../model/gte-multilingual-reranker-base", # retrieve阶段的reranker预训练模型路径
        ret_reranker_q_t_lora_path = "../../model/lora_question_to_table_reranker", # question to table的reranker lora路径
        
        max_retriever_tokens = 4096, # 生成用于检索向量时的最大token数
        max_ret_reranker_tokens = 4096, # retrieve阶段的reranker的最大token数
        
        question_to_table_k = 20, # 用question检索table时召回的table数
        cell_to_passage_k = [1,2,3], # 用每个单元格召回的passage数
        
        # row_scorer和reader(共用base model)   
        pretrained_worker_path = "../../model/Qwen2.5-7B-Instruct", # base model的路径               
        use_row_scorer = True, # 设置是否使用行row_scorer进行行筛选，如果不使用，那么全部行作为reader的输入

        row_scorer_lora_path = "../../model/lora_row_scorer", # row_scorer的lora路径
        max_row_scorer_tokens = 4096, # row_scorer模型的截断token数
        max_row_num = 1, # row_scorer选取的最大的行数
        
        reader_lora_path = "../../model/lora_reader", # reader的lora路径
        max_reader_tokens = 8192, # reader截断token数
        max_answer_tokens = 120, # answer-text的最大token数
                      
        # results保存路径
        result_dev_path = "../../ottqa_data/results/dev.json", # dev集结果路径
        result_test_path = "../../ottqa_data/results/test.json", # test集结果路径
        ):
        
        # settings
        self.rand_seed_value = rand_seed_value
            
        # 要评估的子集    
        self.eval_dataset = eval_dataset
        
        # question路径  
        self.devset_path = devset_path
        self.testset_path = testset_path

        # tables路径
        self.tables_path = tables_path
        self.table_faiss_path = table_faiss_path

        # passages路径
        self.passages_path = passages_path
        self.passage_faiss_path = passage_faiss_path 
        
        # early fusion路径
        self.early_fusion_path = early_fusion_path
        
        # retriever
        self.retriever_worker_path = retriever_worker_path
        self.retriever_pooling = retriever_pooling           
        self.retriever_q_t_lora_path = retriever_q_t_lora_path
        self.max_retriever_tokens = max_retriever_tokens
        
        # reranker
        self.ret_reranker_worker_path = ret_reranker_worker_path
        self.ret_reranker_q_t_lora_path = ret_reranker_q_t_lora_path
        self.max_ret_reranker_tokens = max_ret_reranker_tokens

        # 用question检索table时召回的table数 
        self.question_to_table_k = question_to_table_k
        
        # 用每个单元格召回的passage数
        self.cell_to_passage_k = cell_to_passage_k
        
        # row_scorer和reader(共用base model)
        self.pretrained_worker_path = pretrained_worker_path
        self.use_row_scorer = use_row_scorer
        
        if self.use_row_scorer == True:
            self.row_scorer_lora_path = row_scorer_lora_path
            self.max_row_scorer_tokens = max_row_scorer_tokens
            self.max_row_num = max_row_num            
        
        self.reader_lora_path = reader_lora_path
        self.max_reader_tokens = max_reader_tokens
        self.max_answer_tokens = max_answer_tokens
            
        # results保存路径
        self.result_dev_path = result_dev_path
        self.result_test_path = result_test_path

        # 确定要评估的数据集
        if eval_dataset == "dev":
            self.dataset_path = self.devset_path
            self.result_path = self.result_dev_path
        elif eval_dataset == "test":      
            self.dataset_path = self.testset_path
            self.result_path = self.result_test_path
        
        # 设定随机数
        self.set_random_seeds()
    
    @staticmethod
    def set_random_seeds(seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        
    @staticmethod
    def strlist_to_vectors(tokenizer,model,model_pooling,strlist,max_length):
        batch_inputs = tokenizer(
            strlist,
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",  
            )
        batch_inputs = {key: val.to(device) for key, val in batch_inputs.items()}
        if model.training == False:
            with torch.no_grad():
                batch_outputs = model(**batch_inputs)
                if model_pooling == "mean":
                    batch_vectors = get_mean_pooling_vec(batch_outputs.last_hidden_state,batch_inputs["attention_mask"])  
                elif model_pooling == "last":
                    batch_vectors = last_token_pool(batch_outputs.last_hidden_state,batch_inputs["attention_mask"])
                elif model_pooling == "first": 
                    batch_vectors = batch_outputs.last_hidden_state[:, 0]
                    batch_vectors = F.normalize(batch_vectors, p=2, dim=1) 
                else:
                    print("Wrong pooling setting!")
                    sys.exit()
        else:
            batch_outputs = model(**batch_inputs)
            if model_pooling == "mean":
                batch_vectors = get_mean_pooling_vec(batch_outputs.last_hidden_state,batch_inputs["attention_mask"])  
            elif model_pooling == "last":
                batch_vectors = last_token_pool(batch_outputs.last_hidden_state,batch_inputs["attention_mask"])
            elif model_pooling == "first": 
                batch_vectors = batch_outputs.last_hidden_state[:, 0]
                batch_vectors = F.normalize(batch_vectors, p=2, dim=1) 
            else:
                print("Wrong pooling setting!")
                sys.exit() 
        return batch_vectors     
        
    def get_retriever_base_model(self):
        """获取retriever的base model"""
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(get_abs_path(self.retriever_worker_path),trust_remote_code=True)
        self.retriever_model = AutoModel.from_pretrained(get_abs_path(self.retriever_worker_path),trust_remote_code = True).to(device)
        
    def get_table_retriever(self):
        """获取以整个table为检索对象的retriever"""
        self.table_retriever = FaissRetriever(
                                tokenizer = self.retriever_tokenizer,
                                pretrained = self.retriever_model,
                                lora_path = self.retriever_q_t_lora_path, 
                                corpusbase_or_path = self.table_faiss_path,
                                model_pooling = self.retriever_pooling,
                                )    

    def get_ret_reranker_base_model(self):
        """获取retrieve阶段的reranker的base model"""
        self.ret_reranker_tokenizer = AutoTokenizer.from_pretrained(get_abs_path(self.ret_reranker_worker_path),trust_remote_code=True)
        self.ret_reranker_model = AutoModelForSequenceClassification.from_pretrained(get_abs_path(self.ret_reranker_worker_path),trust_remote_code = True).to(device)

    def get_ret_table_reranker(self):
        """获取以整个table为检索对象的reranker"""
        self.ret_table_reranker = RetReranker(tokenizer=self.ret_reranker_tokenizer,
                                       pretrained=self.ret_reranker_model,
                                       lora_path=self.ret_reranker_q_t_lora_path)
   
    def get_row_scorer_reader_base_model(self):
        """获取row_scorer和reader的base model"""
        self.tokenizer = AutoTokenizer.from_pretrained(get_abs_path(self.pretrained_worker_path),trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(get_abs_path(self.pretrained_worker_path),torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True)
    
    def get_row_scorer(self):
        """获取row_scorer"""
        self.row_scorer = LoraWorker(tokenizer=self.tokenizer,
                                pretrained=self.model,
                                lora_path=self.row_scorer_lora_path)
        
    def get_reader(self):
        """获取reader"""
        self.reader = LoraWorker(tokenizer=self.tokenizer,
                                pretrained=self.model,
                                lora_path=self.reader_lora_path)        
                
    def get_resource(self):
        """获取需要的tables、passages、models"""
        # 获取tables
        self.tables = load_json(self.tables_path)
        
        # 获取passages
        self.passages = load_json(self.passages_path)
        
        # 获取retrievers
        self.get_retriever_base_model()
        self.get_table_retriever()
        
        # 获取ret_reranker
        self.get_ret_reranker_base_model()
        self.get_ret_table_reranker()
        self.ret_table_reranker.get_model() 
        
        # 获取row_scorer和reader模型
        self.get_row_scorer_reader_base_model()
        if self.use_row_scorer == True:
            self.get_row_scorer()
        self.get_reader()
        
        # 获取early_fusion的结果，包含dev和test集的问题检索到的可能表格离线连接的段落
        self.dev_test_rerieved_tables = load_pickle(self.early_fusion_path)
        
    def get_row_scorer_result(self, question: str, table: "OttqaTable"):
        # 获取模型
        tokenizer = self.tokenizer
        row_scorer = self.row_scorer.get_model() 
        row_scorer.eval()
        softmax = nn.Softmax(dim=0)

        row_scorer_selected_rows = {}
        for cell_to_passage_num in self.cell_to_passage_k:
            # 处理行
            all_row_num = [i for i in range(len(table.data))]  # 所有行列表
            row_scorer_result = []

            for row_num in all_row_num:  # 每次处理一行
                item_row_nums = [row_num]

                formatted_multi_rows = table.get_multi_row_table_with_passage(
                    self,
                    row_nums=item_row_nums,
                    cell_passage_num=cell_to_passage_num
                )

                row_scorer_prompt = MY_PROMPT["row_scorer"] + MY_PROMPT["prompt_3"] + question + "\n\n" + formatted_multi_rows + "\n"
                row_scorer_prompt = row_scorer_prompt + MY_PROMPT["prompt_4"] + "\n" + "Your predicted category:"

                input_ids = tokenizer.encode(
                    text=row_scorer_prompt,
                    add_special_tokens=True
                )

                if len(input_ids) > (self.max_row_scorer_tokens - 200):
                    row_scorer_prompt = tokenizer.decode(
                        input_ids[:(self.max_row_scorer_tokens - 200)]
                    ) + "\n\n" + MY_PROMPT["prompt_4"] + "\n" + "Your predicted category:"

                    input_ids = tokenizer.encode(
                        text=row_scorer_prompt,
                        add_special_tokens=True
                    )

                input_ids.append(tokenizer.pad_token_id) 
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = row_scorer(input_ids=input_ids)
                    logits = torch.squeeze(output['logits'])[-2]
                    category = tokenizer.decode(torch.argmax(logits))
                    probs = softmax(logits).tolist()

                prob_0 = probs[
                    tokenizer.encode("0", add_special_tokens=False)[0]
                ]
                prob_1 = probs[
                    tokenizer.encode("1", add_special_tokens=False)[0]
                ]

                row_scorer_result.append({
                    "row_num": row_num,
                    "category": category,
                    "prob_0": prob_0,
                    "prob_1": prob_1
                })

            sorted_row_scorer_result = sorted(
                row_scorer_result,
                key=lambda x: x['prob_1'],
                reverse=True
            )

            selected_rows = []

            for i in range(len(sorted_row_scorer_result)):
                if sorted_row_scorer_result[i]['prob_1'] > sorted_row_scorer_result[i]['prob_0']:
                    selected_rows.append(sorted_row_scorer_result[i])

                if len(selected_rows) >= self.max_row_num:
                    break

            if len(selected_rows) == 0:
                selected_rows = sorted_row_scorer_result[:self.max_row_num]

            row_scorer_selected_rows[cell_to_passage_num] = selected_rows

        return row_scorer_selected_rows  

    def get_reader_answer(self,question:"OttqaQuestion",table:"OttqaTable",row_scorer_selected_rows):
        result_item = {}
        tokenizer = self.tokenizer
        reader = self.reader.get_model()
        reader.eval()

        current_avg_log_prob = -1000
        avg_log_prob_item = {}
        raw_log_prob_item = {}
        gnmt_score_item = {}
        reader_answer_item = {}
        for cell_to_passage_num in self.cell_to_passage_k:
            # title和Introduction等通用信息
            reader_prompt =  MY_PROMPT["prompt_0"] + question.question + "\n\n" + MY_PROMPT["prompt_1"]  + "\n"
            reader_prompt = reader_prompt + table.format_table_with_passage(self, row_nums=[sel_item["row_num"] for sel_item in row_scorer_selected_rows[cell_to_passage_num]],cell_passage_num = cell_to_passage_num)
            reader_prompt = reader_prompt + MY_PROMPT["prompt_2"]
            encoded_prompt = tokenizer.encode(reader_prompt,add_special_tokens = True) 
            if len(encoded_prompt) > (self.max_reader_tokens-200): 
                reader_prompt = tokenizer.decode(encoded_prompt[0:(self.max_reader_tokens-200)]) + "\n\n" + MY_PROMPT["prompt_2"] 
            if "Qwen" in self.pretrained_worker_path: 
                reader_answer,logits,avg_log_prob,raw_log_prob, gnmt_score = chat_qwen(reader,tokenizer,reader_prompt,self.max_answer_tokens) 
            
            # 记录avg_log_prob,raw_log_prob, gnmt_score的分数和不同cell_to_passage_num的answer
            avg_log_prob_item[cell_to_passage_num] = avg_log_prob
            raw_log_prob_item[cell_to_passage_num] = raw_log_prob
            gnmt_score_item[cell_to_passage_num] = gnmt_score
            reader_answer_item[cell_to_passage_num] = reader_answer        
            
            # 运行时reader_annswer按avg_log_prob确定，如需根据raw_log_prob, gnmt_score确定，需要对结果进行后处理
            if avg_log_prob > current_avg_log_prob:
                current_avg_log_prob = avg_log_prob
                current_reader_answer = reader_answer

        
        #计算指标
        result_item["question_id"] = question.question_id
        if hasattr(question, 'table_id'):
            result_item["gold_table_id"] = question.table_id
        result_item["retrieved_table_id"] = table.uid
        result_item["row_scorer_selected_rows"] = row_scorer_selected_rows
        result_item["reader_confidence"] = current_avg_log_prob
        result_item["predicted_answer"] = current_reader_answer
        # 记录每个cell_to_passage_num的avg_log_prob,raw_log_prob, gnmt_score的分数和不同cell_to_passage_num的answer
        result_item["avg_log_prob"] = avg_log_prob_item
        result_item["raw_log_prob"] = raw_log_prob_item
        result_item["gnmt_score"] = gnmt_score_item
        result_item["candidate_answer"] = reader_answer_item
        
        if hasattr(question, 'answer-text'): #对于dev集，存在gold_answer，计算指标，指标也在result_item中返回
            result_item["gold_answer"] = question.__dict__['answer-text']
            exact_match = compute_exact(result_item["gold_answer"],result_item["predicted_answer"])
            f1 = compute_f1(result_item["gold_answer"],result_item["predicted_answer"])
            result_item["exact_match"] = float(exact_match)
            result_item["f1"] = f1

        return result_item

class LoraWorker:
    """需要加载lora weights工作的模型"""
    def __init__(self,tokenizer,pretrained,lora_path=None):
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.lora_path = lora_path
    
    def get_model(self):
        if self.lora_path == None:
            self.model = self.pretrained
        else:
            lora_path = get_abs_path(self.lora_path)
            self.model = PeftModel.from_pretrained(self.pretrained,lora_path,trust_remote_code=True).to(device)
        return self.model
    
class FaissIndex:
    """向量库index的类"""
        
class FaissRetriever(LoraWorker):
    """retriever的类"""
    def __init__(self,tokenizer,pretrained,corpusbase_or_path,model_pooling,lora_path):
        # 获取模型
        super().__init__(tokenizer=tokenizer,pretrained=pretrained,lora_path=lora_path)
        self.model_pooling = model_pooling
        # 获取index路径
        self.corpusbase_or_path = corpusbase_or_path # corpusbase_or_path是一个index变量或是faissindex文件的地址
        # 加载index
        self.corpusbase = load_pickle(corpusbase_or_path)
        # 将index从CPU迁移到GPU
        res = faiss.StandardGpuResources()
        self.corpusbase.index = faiss.index_cpu_to_gpu(res, 0, self.corpusbase.index)
    
    def do_retrieve(self, batch_query: List[str], k: int, max_length=8192):
        batch_vectors = OttqaSolver.strlist_to_vectors(self.tokenizer,self.model,self.model_pooling,batch_query,max_length)
        query_vectors = np.array(batch_vectors.tolist()).astype('float32')
        D, I = self.corpusbase.index.search(query_vectors, k)
        return D,I,batch_vectors

class RetReranker(LoraWorker):
    def __init__(self,tokenizer,pretrained,lora_path=None):
        super().__init__(tokenizer=tokenizer,pretrained=pretrained,lora_path=lora_path)
        
    def do_ret_rerank(self, query:str, candidate_context:List[str]):
        pairs = []
        for context in candidate_context:
            pairs.append([query,context])
        
        reranked_score = []
        for pair in pairs:        
            if self.model.training == False:
                with torch.no_grad():
                    inputs = self.tokenizer([pair], padding=True, truncation=True, return_tensors='pt', max_length=4096)
                    inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
                    score = self.model(**inputs, return_dict=True).logits.view(-1, )
                    
            else:
                inputs = self.tokenizer([pair], padding=True, truncation=True, return_tensors='pt', max_length=4096)
                inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
                score = self.model(**inputs, return_dict=True).logits.view(-1, )
            
            reranked_score.append(score.item())
        
        return reranked_score
    
class OttqaQuestion:
    """表示一个问题的类"""
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def build_question_to_table_prompt(self):
        question_to_table_prompt = "Now we have a question: " + self.question + "\n\n" + "Please retrieve the Wikipedia table that provides essential information for answering the question."
        self.question_to_table_prompt = question_to_table_prompt
        return question_to_table_prompt
        
class OttqaTable:
    """表示一个table的类"""
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
     
    def build_common_introduction(self):
        """格式化表格的说明部分"""
        common_introduction = "Title:" + self.title + "\n"
        common_introduction = common_introduction + "Introduction:" + self.intro + "\n"
        common_introduction = common_introduction + "Section title:" + self.section_title + "\n"
        self.common_introduction = common_introduction
        return common_introduction
        
    def get_multi_row_table(self,row_nums = None):
        """
        构造表头和数据行,入参是行组成的列表
        """
        # row_nums没有输入时就取所有行
        if row_nums == None:
            row_nums = [i for i in range(len(self.data))]
        #表头
        common_header = "Table:" + "\n" + "|"
        for header in self.header:
            if isinstance(self.header[0], list): # 单元格带超链接
                common_header = common_header + header[0] + "|"
            else: # 单元格无超链接
                common_header = common_header + header + "|"
        common_header = common_header + "\n"
        #表头与数据行分隔线
        for _ in range(len(self.header)):
            common_header = common_header + "|---"
        common_header = common_header + "|\n"
        
        multi_row_table = common_header
        #处理多行表格
        for row_num in row_nums:
            row = self.data[row_num]
            multi_row_table = multi_row_table + "|"
            for cell in row:
                if isinstance(cell, list): # 单元格带超链接
                    multi_row_table = multi_row_table + cell[0] + "|"
                else: # 单元格无超链接
                    multi_row_table = multi_row_table + cell + "|"
            multi_row_table = multi_row_table + "\n"
        
        return multi_row_table
  
    def get_multi_row_passage(self,solver:OttqaSolver,row_nums = None,cell_passage_num = 5):
        if row_nums == None:
            row_nums = [i for i in range(len(self.data))]
        multi_row_passage = "Hyperlinked passages:\n"
        passage_ids = []
        for row_num in row_nums:
            for col_num in range(len(self.data[0])):
                for passage_id in self.fused_passage_ids[row_num][col_num][:cell_passage_num]:
                    if passage_id not in passage_ids:
                        passage_ids.append(passage_id)
        for passage_id in passage_ids:
            if solver.passages[passage_id] != '':
                multi_row_passage = multi_row_passage + solver.passages[passage_id] + "\n"
        return multi_row_passage          
        
    def format_table(self,row_nums = None):
        """build_common_introduction + get_multi_row_table,获取带说明的表格"""
        # 表格周围的说明信息
        common_introduction = self.build_common_introduction()
        # 表格体部分
        multi_row_table = self.get_multi_row_table(row_nums=row_nums)
        # 周围说明和表格体拼接
        formatted_table = (common_introduction + "\n" + multi_row_table).rstrip('\n')
        return formatted_table
    
    def format_table_with_passage(self,solver:OttqaSolver,row_nums = None,cell_passage_num = 5):
        """build_common_introduction + get_multi_row_table + get_multi_row_passage,获取带说明并计入段落的表格"""
        # 获取周围说明和表格体
        formatted_table = self.format_table(row_nums = row_nums)
        # 获取段落
        multi_row_passage = self.get_multi_row_passage(solver,row_nums = row_nums,cell_passage_num = cell_passage_num)
        # 拼接
        formatted_table_with_passage = formatted_table + "\n" + multi_row_passage
        return formatted_table_with_passage
    
    def get_multi_row_table_with_passage(self,solver:OttqaSolver,row_nums = None,cell_passage_num = 5):
        """get_multi_row_table + get_multi_row_passage,获取不带说明但计入段落的表格"""
        # 表格体部分
        multi_row_table = self.get_multi_row_table(row_nums=row_nums)
        # 段落部分
        multi_row_passage = self.get_multi_row_passage(solver,row_nums = row_nums,cell_passage_num = cell_passage_num)
        # 拼接
        multi_row_table_with_passage = multi_row_table + "\n" +  multi_row_passage
        return multi_row_table_with_passage