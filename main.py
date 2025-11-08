import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import re
from openai import OpenAI
from prompt import *
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM




# API adoption, input prompt and get answer in steps
def get_message(prompt, temperature):
    try:
        client = OpenAI(
            api_key = "xxxxx", # personal api key here
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1" # website url and targeted model
        )

        completion = client.chat.completion.create(
            model = "qwen_plus",
            message = [
                {"role": "system", "content": "You are a helpful assistatn."},
                {"role": "user", " content": prompt},
            ]
            temperature = temperature
        )
        return completion.choice[0].message.content

    except Exception as e:
        print(f"wrong info: {e}")


# split answers with "steps":: into parts
def split_steps(input_str):
    steps = re.split(r'(?=step \d', input_str)
    steps = [step.strip() for step in steps if step]
    return steps

def mock_policy_model_generate(problem_text,
                               num_solutions = 6 ) -> List[List[str]] :
    
    solutions = []

    # multiple soluctions
    for _ in range(num_solutions):
        temperatures = [i * 0.1 for i in range(5, 13)]
        # random choose temper
        temperature = random.choice(temperatures)
        # get response from LLM
        problem_text = policy_model_generate_template.format(problem_text = problem_text)
        solution = get_message(problem_text, temperature = temperature)
        
        # solution split into List[str]
        steps = split_steps(solution)
        # List[List[str]], 8 solution(list of steps) in 1 list
        solutions.append(steps) 


"""
“每一步都让模型蒙特卡洛式地自我验证 8 次，看它是否在该步之后仍能持续走向正确答案。”
如果在某个推理步骤之后，模型多数时候都能生成正确的最终答案，则说明这个步骤是「潜在正确的」。
"""

def evaluate_consistency(result):
    res_arr = np.array(result) # list -> array
    mean_res = np.mean(res_arr)
    std_res = np.std(res_arr)
    variacne = np.var(res_arr)
    if std_res < 0.1 * mean_res:
        return 1
    return 0

def mock_mc_estimation_constrained(question,
                                   solution_steps, continue_answer_prompt):
    """
    once for one solution
    comulative steps as prompt, as model continue to generate for 8 times
    then see whether 8 results are consistent(答案相近)
    return 0 or 1 for each step, 
    so result is like [1,1,1,0,1]
    """

    num_simulations = 8
    accumulated_steps = ""
    result = []

    continue_answer_prompt = continue_answer_prompt

    for i, step in enumerate(solution_steps):
        # 每次都先给问题，然后给出之前生成的步骤，给1步，给2步。。这样给下去，直到给完 一个solution中的所有steps
        # 所以累积 给 已经生成的步数，然后继续生成答案。
        accumulated_steps += "f\n Step {i+1}: {step}"
        # 构建给不同步数的prompt
        current_prompt_text = continue_answer_prompt.format(
            question = question,
            accumulated_steps = accumulated_steps
        )

        answer_lists = []

        for sim_num in range(num_simulations):
            
            # i=1就是给了1步step后，sim_num=1进行第1次仿真
            # model续写，返回 回答
            full_solution = get_message(current_prompt_text)
            # re.DOTALL: 它让 . 可以匹配 包括换行符在内的所有字符。 默认情况下 . 不会匹配 \n
            match = re.search(r"最终答案是:\s*(.*)", full_solution, re.DOTALL)
            # 有答案，就保存
            if match:
                # group(1) 抽取 第1个括号中的匹配对象；假如是0，那就是整个匹配，包括字符串 
                gererated_ans_str = match.group(1).strip()
                try:
                    ans_val = float(gererated_ans_str)
                    answer_lists.append(ans_val)
                
                except ValueError:
                    # 无法转化为浮点数。
                    pass # ignore
            else:
                pass
        
        # 每个 step，生成8次答案，收集，然后做投票
        if len(answer_lists) < 2:
            step_result = 0
        else:
            step_result = evaluate_consistency(answer_lists)
        # 第i步的投票结果，加进来
        result.append(step_result)

    # i step -> i elements
    return result


    

def mc_labeling_for_problem(problem_text) -> List[Dict]:
    """
    [
        {
            "problem": problem_text,
            "steps": [
                {"text": "Step_1....", "mc_labels": 1},
                {"text": "Step_2....", "mc_labels": 0},
                ...
            ]
        },

        .....

    ]
    
    """
    # 将存入的 一个 问题，解答6次，各次按照step split，得到List[List[str]]
    solutions = mock_policy_model_generate(problem_text=problem_text,
                                           num_solutions=6)
    print(solutions)

    result = []
    # 一个solution，包含多个step的str的list。每次处理一个solution
    for solution in solutions:
        mc_labels = mock_mc_estimation_constrained(problem_text, solution)
        step_data = []
        # here, solution as a list of steps, mc as steps' labels
        # same numbers of elements of 2 lists
        for step_text, label in zip(solution, mc_labels):
            step_data.append({
                "text": step_text,
                "mc_label": label
            })
        # after append of each step in one solution, append
        result.append(
            {
            "problem": problem_text,
            "steps": step_data
            }
        )
    return result
    






#####################LLM judges each step alone#######################

def mock_llm_judge(question: str, step_text: str) -> int:
    """
    llm as a judge
    
    """
    prompt = eval_prompt(question, step_text)
    response = get_message(eval_prompt)
    right_or_not = response.split("正确性")[-1] # 0 or 1
    return right_or_not


# 有点怪，对solutioin中 每一个step 单独进行 判断是否合理，
# 没有前边步骤，没有后边步骤，只对 中间 一个孤零零的步骤 进行判断

def llm_juedge_labeling(question_data: List[Dict]):
    """
    [
        {
            "problem": problem_text,
            "steps": [
                {"text": "Step_1....", "mc_labels": 1, "judge_label": 1},
                {"text": "Step_2....", "mc_labels": 0, "judge_label": 0},
                ...
            ]
        },

        .....
        
    ]
    
    """

    # 对于每个question 的solution 的dict
    output = []
    for solution_dict in question_data:
        new_steps = []
        for step_info in solution_dict["steps"]:
            step_text = step_info["text"]
            llm_juedge_res = mock_llm_judge(solution_dict["problem"], step_text)
            step_info["judge_label"] = llm_juedge_res
            new_steps.append(step_info)
        # 直接进行整体覆盖
        solution_dict["steps"] = new_steps

        # 不在原来的dict上修改，而是重新创造一个，修改，复制
        output.append(solution_dict)
    return output


# double check and only keep double 1s
def consensus_filtering(data: List[Dict]) -> List[Dict]:

    """
    返回数据格式：
    [
        {
            "problem": problem_text,
            "steps": [
                {"text": "Step_1....", "final_label": 0},
                {"text": "Step_2....", "final_label": 1},
                ...
            ]
        },

        .....

    ]
    """
    
    filtered_data = []
    for sol_dict in data:
        correct_steps = []
        for step_info in sol_dict["steps"]:
            judge_label_val = step_info.get("judge_label", 0)
            mc_label_val = step_info.get("mc_label", 0)
            
            final_label = (judge_label_val and mc_label_val)

            correct_steps.append(
                {
                    "text": step_info["text"],
                    "final_label": 1
                }
            )
        filtered_data.append(
            {
                "problem": sol_dict["problem"],
                "steps": correct_steps
            }
        )
    return filtered_data





class PRMDataset(Dataset):
    """
        所有solution中的所有step都放在了一起，平面化了
        [
        (embed_question1_step1, label_1 = 0),
        (embed_quesiton1_step2, label_2 = 1),
        .........
        (embed_question2_step1, label_1 = 1),
        ......
        ]
    """
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        for solution in data:
            for step_info in solution["steps"]:
                step_text = step_info["text"]
                label = step_info["final_label"]
                embedding = tokenizer([step_text])

                # 对于每个
                self.samples.append((embedding, label))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emb, label = self.samples[idx]
        return emb, label


class SimplePRM(nn.Module):
    """
    easy model example of PRM(process reward model)
    对于每一步，返回（0,1）之间的数，表示这一步正确的概率-----具体如何做，还是有点疑惑
    """
    def __init__(self, model):
        super().__init__()
        self.net = model
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        logits = self.net(x)
        scores = self.sigmoid(logits)
        return scores


# 训练PRM模型，输入 每一步的embedding，输出 分数（二分类）
def train_prm_model(filtered_data,
                    epochs = 3,
                    batch_size = 8,
                    model = None):
    dataset = PRMDataset(filtered_data)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True)
    model = SimplePRM() # embed -> score
    # 二分类 损失
    criterion = nn.BECLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(),
                           lr = 1e-5)

    model.train()
    # 几轮
    for epoch in range(epochs):
        total_loss = 0.0
        # 抽取数据, 注意这里有batch，所以label是一个batch中的所有label，不止一条数据
        for emb, label in dataloader:
            optimizer.zero_grad()
            scores = model(emb)  # [batch_size, 1]
            label = label.float().unsqueeze(1) # [batch_size, 1]
            # loss
            loss = criterion(scores, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # 每个epoch 输出一次总体loss的平均值(基于batch的)
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss = {avg_loss:.4f}")
    return model


def evaluate_best_of_n(model,
                       problem_text,
                       n = 8,
                       tokenizer = None):
    # generate 8 solution split into steps in a list
    solutions = mock_policy_model_generate(problem_text,
                                           num_solutions = n)

    best_score = -1.0
    best_sol = None
    
    # for each solution containing steps' list
    for solution in solutions:
        step_embeddings =  [tokenizer([s]) for s in solution]
        step_scores = []
        # 每一步 打分
        for emb in step_embeddings:

            emb = emb.unsqueeze(0) # shape=[1,16]

            with torch.no_grad():
                score = model(emb) # shape=[1,1]
                step_scores.append(score.item())

        # 简单策略是 "product" 策略，也可以改成 "min" 或 "last"，论文中有提到这几种组合方法
        import math
        product_score = 1.0
        for score in step_scores:
            # 这里直接 将 各个步骤的分数 相乘作为 句子的分数
            product_score *= score
        
        # still in the loop. filter the best solution with highest score
        if product_score > best_score:
            best_score = product_score
            best_solution =  solution
    
    print(f"[DEBUG] Best-of-{n} Score = {best_score:.5f}, solution = {best_sol}")
    # 返回分数最高的solution
    return best_solution


def evaluate_processbench_style(model: SimplePRM, steps: List[str], tokenizer=None) -> int:
    """
    模拟 PROCESSBENCH 风格的步骤级评估：
    给定一个完整的推理过程（多步），我们要求找出第一步出错的下标。
    如果模型判断所有步骤皆为正确，则返回 -1 表示全对。

    embedding + PRM打分，当分数<0.5时，就视为错误。
    """

    for idx,  step_text in enumerate(steps):
        emb = tokenizer.encode([step_text])
        with torch.no_grad():
            score = model(emb).item()
        
        if score < 0.5:
            return idx
    
    return -1


def main_demo():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct", trust_remote_code=True)

    # 一个题目
    input_file = open("xxx", "r")
    filter_datas = []
    for problem_text in input_file.readlines():
        
        # 概率模型 + MC估计构建数据集
        mc_data = mc_labeling_for_problem(problem_text)

        # LLM-as-a-judge 评估
        judge_data = llm_juedge_labeling(mc_data)

        filtered_data = consensus_filtering(judge_data)
        filtered_data.append(filtered_data)
    # train PRM
    print("PRM begins...")
    prm_model = train_prm_model(filtered_data,
                                epochs = 3,
                                batch_size = 4,
                                model = model)
    print("\n=== Best-of-N ===")
    evaluate_best_of_n(prm_model, problem_text, n=8, tokenizer=tokenizer)

    print("\n=== PROCESSBENCH 风格测试 ===")
    test_steps =["Step1", "Step2", "Step3", "Step4",]
    err_idx = evaluate_processbench_style(prm_model, test_steps, tokenizer=tokenizer)
    if err_idx == -1:
        print("PRM 认为所有步骤都正确（在此示例中只是一个构造的）。")
    else:
        print(f"PRM 认为第 {err_idx} 步是第一个错误的步骤。")

if __name__ == "__main__":
    main_demo()








