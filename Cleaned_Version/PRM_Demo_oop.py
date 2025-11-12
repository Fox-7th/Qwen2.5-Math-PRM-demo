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
            completion = self.client.chat.completions.create(
                model = self.model_name,
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature = temperature
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error info: {e}")

llm = LLM_Responser(api_key = "xxx")
response = llm("告诉我今天是几月几日？")



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
        self.num_solutions = num_solutions
        self.prompt_template = continue_answer_prompt
        self.num_simulations = 8

    def evaluate_consistency(self, result):
        res_arr = np.array(result) # list -> array
        mean_res = np.mean(res_arr)
        std_res = np.std(res_arr)
        if std_res < 0.1 * mean_res:
            return 1
        return 0

    # consistency estimation
    def evaluate_step_consistency(
                self,
                solution_steps: List, 
                question, 
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
            accumulated_steps += f"\n Step {i+1}: {step}"
            current_prompt_text = self.prompt_template.format(
                question = question,
                accumulated_steps = accumulated_steps
            )
            answer_lists = []

            for sim_num in range(self.num_simulations):
                full_solution = self.llm_responder(current_prompt_text)   
                match = re.search(r"最终答案是:\s*(.*)", full_solution, re.DOTALL)

                if match:
                    generated_ans_str = match.group(1).strip()
                    try:
                        ans_val = float(generated_ans_str)
                        answer_lists.append(ans_val)
                    except ValueError:
                        pass # ignore
                else:
                    pass
            
            if len(answer_lists) < 2:
                step_result = 0
            else:
                step_result = self.evaluate_consistency(answer_lists)
            result.append(step_result)
        # i step -> i elements
        return result


    def label_problem(self, question: str, generator: Answer_Generator_Step_Splitter) -> List[Dict]:
        """
        return format:
        [
            {
                "problem": question,
                "steps": [{"text": ..., "mc_label": ...}, ...]
            },
            ...
        ]
        """
        all_solutions = generator.generate_solution_steps(question)
        labeled_solutions = []

        for solution_steps in all_solutions:
            mc_labels = self.evaluate_step_consistency(solution_steps, question)
            step_dicts = []
            for step_text, label in zip(solution_steps, mc_labels):
                step_dicts.append({
                    "text": step_text,
                    "mc_label": int(label)
                })

            labeled_solutions.append({
                "problem": question,
                "steps": step_dicts
            })

        return labeled_solutions



class LLM_StepJudge:
    def __init__(self, 
                llm, 
                eval_prompt = eval_prompt, 
                ):
        self.llm = llm
        self.eval_prompt = eval_prompt

    def judge_step(self, 
                   question,
                   solution_steps):
        step_eval_list = []
        for step in solution_steps:
            prompt = self.eval_prompt.format(question = question, solution_steps = step)
            response = self.llm(prompt)
            right_or_not = response.split("正确性")[-1] # 0 or 1
            step_eval_list.append(right_or_not)
        return step_eval_list

    def label_problem(self, mc_data: List[Dict]) -> List[Dict]:
        """
        return format:
        [
            {
                "problem": question,
                "steps": [{"text": ..., "j_label": ...}, ...]
            },
            ...
        ]
        """
        judged_data = []
        for item in mc_data:
            question = item["problem"]
            step_texts = [s["text"] for s in item["steps"]]
            judge_labels = self.judge_step(question, step_texts)

            new_steps = []
            for step, j_label in zip(item["steps"], judge_labels):
                new_steps.append({
                    "text": step["text"],
                    "judge_label": int(j_label)
                })

            judged_data.append({
                "problem": question,
                "steps": new_steps
            })

        return judged_data



class LabelMerger:
    def merge(self, mc_data: List[Dict], judge_data: List[Dict]) -> List[Dict]:
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

    def consensus_filter(self, merged_data: List[Dict]) -> List[Dict]:
        filtered_data = []
        for sol_dict in merged_data:
            correct_steps = []

            for step_info in sol_dict["steps"]:
                final_label = int(step_info["mc_label"] and step_info["judge_label"])
                correct_steps.append({
                    "text": step_info["text"],
                    "final_label": final_label
                })
            filtered_data.append({
                "problem": sol_dict["problem"],
                "steps": correct_steps
            })

        return filtered_data



class SimplePRM(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class PRMDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer):
        self.samples = []
        for item in data:
            for step in item["steps"]:
                text = step["text"]
                label = step["final_label"]

                # embedding-to-be-decided depending on needs
                emb = tokenizer([text], return_tensors="pt")["input_ids"].float().mean(dim=1).squeeze(0)
                self.samples.append((emb, torch.tensor(label, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class PRMTrainer:
    def __init__(self, tokenizer, model=None):
        self.tokenizer = tokenizer
        self.model = model if model else SimplePRM(input_dim=768)

    def build_dataset(self, filtered_data: List[Dict]) -> Dataset:
        return PRMDataset(filtered_data, self.tokenizer)

    def train(self, dataset: Dataset, epochs=3, batch_size=8):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for emb, label in dataloader:
                optimizer.zero_grad()
                scores = self.model(emb.unsqueeze(0)).item()
                label = label.float().unsqueeze(1)
                loss = criterion(scores, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"[PRM] Epoch {epoch+1}, Loss = {avg_loss:.4f}")

    def evaluate_best_of_n(self, question: str, generator: Answer_Generator_Step_Splitter, n=8):
        best_score = -1.0
        best_solution = None
        solutions = generator.generate_solution_steps(question)

        for solution in solutions:
            step_embeddings = [self.tokenizer([s]) for s in solution]
            step_scores = []

            for emb in step_embeddings:
                emb = emb.unsqueeze(0)  # [1, D]
                with torch.no_grad():
                    score = self.model(emb)
                    step_scores.append(score.item())

            product_score = 1.0
            for score in step_scores:
                product_score *= score

            if product_score > best_score:
                best_score = product_score
                best_solution = solution

        print(f"[Best-of-{n}] Score: {best_score:.4f}")
        return best_solution

    def evaluate_processbench_style(self, steps: List[str]) -> int:
        for idx, step_text in enumerate(steps):
            emb = self.tokenizer([step_text])['input_ids']
            with torch.no_grad():
                score = self.model(emb).item()
            if score < 0.5:
                return idx
        return -1



class ReasoningEvaluator:
    def __init__(self, llm_responder, tokenizer):
        self.generator = Answer_Generator_Step_Splitter(llm_responder)
        self.mc_evaluator = MonteCarloEvaluator(llm_responder)
        self.llm_judge = LLM_StepJudge(llm_responder, eval_prompt=eval_prompt)
        self.merger = LabelMerger()
        self.tokenizer = tokenizer
        self.prm_trainer = PRMTrainer(tokenizer)

    def run_on_problem(self, question: str):
        # Generate and label with MC
        mc_data = self.mc_evaluator.label_problem(question, self.generator)

        # LLM judge
        judge_data = self.llm_judge.label_problem(mc_data)

        # Merge
        merged = self.merger.merge(mc_data, judge_data)

        # Consensus filter
        filtered = self.merger.consensus_filter(merged)

        # Train PRM
        print("[INFO] Start training PRM...")
        dataset = self.prm_trainer.build_dataset(filtered)
        self.prm_trainer.train(dataset)

        # Evaluation
        print("\n=== Best-of-N Evaluation ===")
        self.prm_trainer.evaluate_best_of_n(question, self.generator)

        print("\n=== PROCESSBENCH 风格测试 ===")
        dummy_steps = ["Step1", "Step2", "Step3", "Step4"]
        err_idx = self.prm_trainer.evaluate_processbench_style(dummy_steps)
        if err_idx == -1:
            print("所有步骤预测为正确。")
        else:
            print(f"PRM 预测第 {err_idx} 步是第一个错误的步骤。")

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")
    llm_responder = LLM_Responser(api_key="your_api_key_here")
    evaluator = ReasoningEvaluator(llm_responder, tokenizer)
    question = "一个苹果加一个苹果等于几？"
    evaluator.run_on_problem(question)


if __name__ == "__main__":
    main()

