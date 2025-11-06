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



# Prompt Templates with Embedded Question
policy_model_generate_template = """
你是一个推理模型，我给你一个Question。你根据题目:每个步骤前都要加 Step x，其中x为步骤编号
要求：
记得加step x
每个推理步骤应当详细而且便于理解。

问题如下: {Question}.
"""

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

def mock_policy_model_generate(problem_test,
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
        # List[List[str]]
        solutions.append(steps) 








"""
“每一步都让模型蒙特卡洛式地自我验证 8 次，看它是否在该步之后仍能持续走向正确答案。”
如果在某个推理步骤之后，模型多数时候都能生成正确的最终答案，则说明这个步骤是「潜在正确的」。
"""



def mock_mc_estimation_constrained(question,
                                   solution_steps):
    num_simulations = 8
    accumulated_steps = ""
    result = []

    continue_answer_prompt = """
        给你一个题目，你需要基于已有知识，作答：
        格式是：下一步...下一步..., 最终答案是： 
        下边的是题目，记得按照格式回答： 
        {question} \n
        已有的解题步骤：
        {accumulated_steps} \n
        你的解答：  
    """

    question = ""


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


    
    














