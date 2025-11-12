import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

import re
import math
import random
import numpy as np
from typing import List, Dict

from prompt import *



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
Each step, check 8 times whether it can leads to right answer
Ff after one step, model has a large chance to finally lead to right answer.
    Then this step has a larg chance to be right and effiecient(leads to right answer)
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
        accumulated_steps += "f\n Step {i+1}: {step}"
        current_prompt_text = continue_answer_prompt.format(
            question = question,
            accumulated_steps = accumulated_steps
        )
        answer_lists = []

        for sim_num in range(num_simulations):
            # continue to answer with given question and already processed steps
            full_solution = get_message(current_prompt_text)
            match = re.search(r"最终答案是:\s*(.*)", full_solution, re.DOTALL)
            
            # if has answer, store it
            if match:
                # store right one
                gererated_ans_str = match.group(1).strip()
                try:
                    ans_val = float(gererated_ans_str)
                    answer_lists.append(ans_val)
                
                except ValueError:
                    pass # ignore
            else:
                pass

        if len(answer_lists) < 2:
            step_result = 0
        else:
            step_result = evaluate_consistency(answer_lists)
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
    # one question, gets 6 answer-paths, all split -> List[List[str]]
    solutions = mock_policy_model_generate(problem_text=problem_text,
                                           num_solutions=6)
    print(solutions)
    result = []
    
    # each time, one solution processed
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

    output = []
    for solution_dict in question_data:
        new_steps = []
        for step_info in solution_dict["steps"]:
            step_text = step_info["text"]
            llm_juedge_res = mock_llm_judge(solution_dict["problem"], step_text)
            step_info["judge_label"] = llm_juedge_res
            new_steps.append(step_info)

        # store in a new one; rather than edit the original one 
        solution_dict["steps"] = new_steps
        output.append(solution_dict)
    return output


# double check and only keep double 1s
def consensus_filtering(data: List[Dict]) -> List[Dict]:
    """
    返回数据格式：
    format of returned data
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
                self.samples.append((embedding, label))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emb, label = self.samples[idx]
        return emb, label



class SimplePRM(nn.Module):
    """
    easy model example of PRM(process reward model)
    """
    def __init__(self, model):
        super().__init__()
        self.net = model
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        logits = self.net(x)
        scores = self.sigmoid(logits)
        return scores


def train_prm_model(filtered_data,
                    epochs = 3,
                    batch_size = 8,
                    model = None):
    dataset = PRMDataset(filtered_data)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True)
    model = SimplePRM()
    criterion = nn.BECLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr = 1e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for emb, label in dataloader:
            optimizer.zero_grad()
            scores = model(emb)  # [batch_size, 1]
            label = label.float().unsqueeze(1) # [batch_size, 1]
            # loss
            loss = criterion(scores, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
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

        for emb in step_embeddings:
            emb = emb.unsqueeze(0) # shape=[1,16]
            with torch.no_grad():
                score = model(emb) # shape=[1,1]
                step_scores.append(score.item())

        product_score = 1.0
        for score in step_scores:
            product_score *= score
        
        # still in the loop. filter the best solution with highest score
        if product_score > best_score:
            best_score = product_score
            best_solution =  solution
    
    print(f"[DEBUG] Best-of-{n} Score = {best_score:.5f}, solution = {best_sol}")
    return best_solution


def evaluate_processbench_style(model: SimplePRM, steps: List[str], tokenizer=None) -> int:

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
    input_file = open("xxx", "r")
    filter_datas = []
    for problem_text in input_file.readlines():
        mc_data = mc_labeling_for_problem(problem_text)
        # LLM-as-a-judge
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
