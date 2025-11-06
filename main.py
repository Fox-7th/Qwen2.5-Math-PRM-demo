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


def mock_mc_estimation_constrained(question,
                                   solution_steps):
    num_simulations = 8
    
    














