import sys,os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
py_dir = os.path.dirname(os.path.realpath(__file__))
utils_path = os.path.abspath(os.path.join(py_dir, "utils"))
sys.path.append(utils_path)
import init_utils

import argparse,logging,time
from datetime import datetime
from tqdm import tqdm

from basic_function import load_json, save_json
from general_settings import setup_logging
from ottqa_pipeline import OttqaQuestion, OttqaSolver, OttqaTable

def main(eval_dataset):
    # 设置日志信息
    setup_logging()
    start_time = datetime.fromtimestamp(time.time())
    solver = OttqaSolver(eval_dataset=eval_dataset)
    logging.info("\n".join(f"{key}:{value}" for key, value in solver.__dict__.items()))
    logging.info(f"start_time:{start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    solver.get_resource()
    dev_test_rerieved_table_keys = set(solver.dev_test_rerieved_tables.keys())

    dataset = load_json(solver.dataset_path)
    table_retriever = solver.table_retriever
    result_dataset, exact_match_score, f1_score = [], [], []

    # 记录循环开始时间
    loop_start_time = datetime.fromtimestamp(time.time())

    for i, data_item in tqdm(enumerate(dataset)):
        # 获取quesiton
        question = OttqaQuestion(**data_item)

        # 构建由question检索table的prompt
        question_to_table_prompt = question.build_question_to_table_prompt()

        # 由question检索table
        table_retriever.get_model()
        D, I, batch_vectors = table_retriever.do_retrieve(
            [question_to_table_prompt], solver.question_to_table_k
        )
        # 获取table
        retrieved_table_ids = [
            table_retriever.corpusbase.index_dict[candi_index] for candi_index in I[0]
        ]
        candidate_context = []
        for retrieved_table_id in retrieved_table_ids:
            table = OttqaTable(**solver.tables[retrieved_table_id])
            formatted_table = table.format_table()
            candidate_context.append(formatted_table)
        reranked_scores = solver.ret_table_reranker.do_ret_rerank(
            question_to_table_prompt, candidate_context
        )
        sorted_pairs = sorted(
            zip(reranked_scores, retrieved_table_ids),
            key=lambda x: x[0],
            reverse=True,
        )
        for table_rank_i in range(solver.question_to_table_k):
            table_id = sorted_pairs[table_rank_i][1]
            if table_id in dev_test_rerieved_table_keys:
                break
        table = OttqaTable(**solver.tables[table_id])

        # 利用离线结果，获取table链接的passages
        num_of_rows, num_of_columns = len(table.data), len(table.data[0])
        fused_passage_ids = [
            [[] for _ in range(num_of_columns)] for _ in range(num_of_rows)
        ]
        for row_num in range(num_of_rows):
            for column_num in range(num_of_columns):
                fused_passage_ids[row_num][column_num] = solver.dev_test_rerieved_tables[
                    table_id
                ]["data"][row_num][column_num][1]
        table.fused_passage_ids = fused_passage_ids

        # 用reranker对table的行进行排序,获取回答问题所需要的行
        if solver.use_row_scorer == True:
            row_scorer_selected_rows = solver.get_row_scorer_result(
                question.question, table
            )
        else:
            row_scorer_selected_rows = None  # 这时OttqaTable.get_multi_row_table会选择所有的行

        # 用row_scorer_selected_rows作为context,获取reader的答案
        result_item = solver.get_reader_answer(
            question, table, row_scorer_selected_rows
        )
        result_dataset.append(result_item)

        # 计算指标
        if hasattr(question, "answer-text"):  # 对于存在答案且计算了指标的情况
            exact_match_score.append(result_item["exact_match"])
            f1_score.append(result_item["f1"])
            if i > 0:
                if i % 20 == 0 or i == len(dataset) - 1:  # 过程输出
                    logging.info(
                        f"num:{i},exact_match:{sum(exact_match_score)/len(exact_match_score):.8f}"
                    )
                    logging.info(
                        f"num:{i},f1:{sum(f1_score)/len(f1_score):.8f}"
                    )
                    save_json(solver.result_path, result_dataset)
        else:  # 对于test集没有标准答案
            if i % 20 == 0 or i == len(dataset) - 1:  # 过程输出
                save_json(solver.result_path, result_dataset)

    # 记录循环结束时间
    loop_end_time = datetime.fromtimestamp(time.time())
    loop_elapsed = loop_end_time - loop_start_time
    end_time = datetime.fromtimestamp(time.time())
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    # 计算并记录时间统计
    total_questions = len(dataset)
    total_loop_time = loop_elapsed.total_seconds()
    avg_time_per_question = (
        total_loop_time / total_questions if total_questions > 0 else 0
    )

    logging.info("=== Time Statistics ===")
    logging.info(
        f"Loop started at: {loop_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logging.info(f"Loop ended at: {loop_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Loop elapsed time: {total_loop_time:.2f} seconds")
    logging.info(f"Number of questions processed: {total_questions}")
    logging.info(f"Average time per question: {avg_time_per_question:.2f} seconds")
    logging.info(
        f"Questions per second: {total_questions/total_loop_time:.2f}"
        if total_loop_time > 0
        else "N/A"
    )

    logging.info(f"end_time:{end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(
        f"Elapsed time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dataset", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.eval_dataset)
