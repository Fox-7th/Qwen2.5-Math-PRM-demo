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

# class that get a question and give an answer
class LLM_Responser:
    def __init__(self, 
                api_key,                 
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                model_name = "qwen_plus"               
                ):
        
        self.model_name = model_name

        self.client = OpenAI(
            api_key = api_key, # personal api key here
            base_url = base_url # website url and targeted model
        )

    
    def __call__(self, prompt: str, temperature = 0.7) -> str:
        try:
            completion = self.client.chat.completion.create(
                model = self.model_name,
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", " content": prompt},
                ],
                temperature = temperature
            )
            return completion.choice[0].message.content
            
        except Exception as e:
            print(f"Error info: {e}")

llm = LLM_Responser(api_key = "xxx")
response = llm("告诉我今天是几月几日？")

# 根据问题，得到多个回答，每个回答切片，step step step，添加
# 得到[[step11, step12, step13], [step21, step22, step23], ...
# 每个step都是具体的一个解题步骤
class Answer_Generator_Step_Splitter:
    def __init__(self,
                 llm_responder: LLM_Responser,
                 prompt_template = policy_model_generate_template,
                 num_solutions=6
                 ):
        self.llm_responder = llm_responder
        self.prompt_template = prompt_template
        self.num_solutions = num_solutions

    # split answers with "steps":: into parts
    def split_step(self, question) -> List[str]:
        steps = re.split(r'(?=step \d', question)
        steps = [step.strip() for step in steps if step]
        return steps 
    
    # 针对一个question生成多个 solution,并且split后，添加。返回[[step1, step2, step3],[step1, step2, stepe],...]
    def generate_solution_steps(self, question: str = "1+1=") -> List[List[str]] :
        steps_solutions = []
        temperatures = [i * 0.1 for i in range(5, 13)]
        # multiple soluctions
        for _ in range(self.num_solutions):
            # random choose temper
            temperature = random.choice(temperatures)
            # get response from LLM
            prompt = self.prompt_template.format(problem_text = question)
            solution = self.llm_responder(prompt, temperature = temperature)
            
            # solution split into List[str]
            steps = self.split_step(solution)
            # List[List[str]], 8 solution(list of steps) in 1 list
            steps_solutions.append(steps) 
        return steps_solutions
        # [[step11, step12, step13], [step21, step22, step23], ...]


    
# splitter = Splitter()


# get answer, split answer into steps, evaluate steps' consistency
class MonteCarloEvaluator:
    def __init__(self,
                llm_responder, 
                # splitter, 
                # question = "1+1=", 
                num_solutions = 8,
                continue_answer_prompt = continue_answer_prompt
                ):
        self.llm_responder =  llm_responder
        # self.question = question
        # self.splitter = splitter
        self.num_solutions = num_solutions
        self.prompt_template = continue_answer_prompt
        self.num_simulations = 8
 
    def evaluate_consistency(self, result):
        res_arr = np.array(result) # list -> array
        mean_res = np.mean(res_arr)
        std_res = np.std(res_arr)
        variacne = np.var(res_arr)
        if std_res < 0.1 * mean_res:
            return 1
        return 0

    # mock_mc_estimation_constrained
    # consistency estimation
    # 这里一次只是  1个问题的1个解决方案
    def evaluate_step_consistency(
                self,
                solution_steps: List, # 一个
                question, #一个
                ):
        """
        once for one solution
        comulative steps as prompt, as model continue to generate for 8 times
        then see whether 8 results are consistent(答案相近)
        return 0 or 1 for each step, 
        so result is like [1,1,1,0,1]
        """

        accumulated_steps = ""
        result = []

        for i, step in enumerate(solution_steps):
            # 每次都先给问题，然后给出之前生成的步骤，给1步，给2步。。这样给下去，直到给完 一个solution中的所有steps
            # 所以累积 给 已经生成的步数，然后继续生成答案。
            accumulated_steps += f"\n Step {i+1}: {step}"
            # 构建给不同步数的prompt
            current_prompt_text = self.prompt_template.format(
                question = question,
                accumulated_steps = accumulated_steps
            )
            answer_lists = []
            for sim_num in range(self.num_simulations):
                
                # i=1就是给了1步step后，sim_num=1进行第1次仿真
                # model续写，返回 回答
                full_solution = self.llm_reponder(current_prompt_text)   


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
                step_result = self.evaluate_consistency(answer_lists)
            # 第i步的投票结果，加进来
            result.append(step_result)

        # i step -> i elements
        return result
        # 一个solution的step list —> consistency list eg.[0 0 0 0 1]


    # gpt给的
    def label_problem(self, question: str, generator: Answer_Generator_Step_Splitter) -> List[Dict]:
        """
        对一个问题生成多个解法，每步打上 mc_label
        返回结构：
        [
            {
                "problem": question,
                "steps": [{"text": ..., "mc_label": ...}, ...]
            },
            ...
        ]
        """
        all_solutions = generator.generate_solution_steps(question) #默认生成6个
        labeled_solutions = []

        for steps in all_solutions:
            mc_labels = self.evaluate_step_consistency(steps, question)
            step_dicts = []
            for step_text, label in zip(steps, mc_labels):
                step_dicts.append({
                    "text": step_text,
                    "mc_label": int(label)
                })

            labeled_solutions.append({
                "problem": question,
                "steps": step_dicts
            })

        return labeled_solutions


# taki in consistency_evaluater


# 对一个问题 多个答案 每一步进行正确与错误的判断，正确给1，错误给0
class LLM_StepJudge:
    def __init__(self, 
                llm, 
                question,
                eval_prompt = eval_prompt, 
                # solution_steps
                ):
        self.llm = llm
        self.eval_prompt = eval_prompt
        self.question = question
        # self.solution_steps = solution_steps

    def judge_step(self, 
                   question,
                   solution_steps):
        step_eval_list = []
        for step in solution_steps:
            prompt = self.eval_prompt.format(question = question, step = step)
            response = self.llm.get_message(prompt)
            right_or_not = response.split("正确性")[-1] # 0 or 1
            step_eval_list.append(right_or_not)

        return step_eval_list


def merge_mc_and_judge(mc_data: List[Dict], judge_data: List[Dict]) -> List[Dict]:
    merged_data = []

    for mc_item, judge_item in zip(mc_data, judge_data):
        assert mc_item["problem"] == judge_item["problem"], "不匹配的问题！"
        merged_steps = []

        for mc_step, judge_step in zip(mc_item["steps"], judge_item["steps"]):
            assert mc_step["text"] == judge_step["text"], "不匹配的 step！"
            merged_steps.append({
                "text": mc_step["text"],
                "mc_label": int(mc_step["mc_label"]),
                "judge_label": int(judge_step["judge_label"])
            })

        merged_data.append({
            "problem": mc_item["problem"],
            "steps": merged_steps
        })

    return merged_data


def mc_labeling_for_problem(problem_text, consistency_evaluater) -> List[Dict]:
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
    solutions = consistency_evaluater.mock_policy_model_generate(problem_text=problem_text,
                                           num_solutions=6)
    print(solutions)

    result = []
    # 一个solution，包含多个step的str的list。每次处理一个solution
    for solution in solutions:
        mc_labels = consistency_evaluater.mock_mc_estimation_constrained(problem_text, solution)
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
