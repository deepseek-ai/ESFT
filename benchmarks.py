import torch
from transformers import pipeline, TextGenerationPipeline
import json
import time
import re

class BaseEvaluator:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.max_new_tokens = config['max_new_tokens']
        self.batch_size = config['eval_batch_size']

    def infer(self, model, tokenizer):
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        input_text = [i['prompt'] for i in self.dataset]

        responses = generator(input_text, max_new_tokens=self.max_new_tokens, do_sample=False, return_full_text=False, temperature=None, top_p=None, batch_size=self.batch_size)

        output = [{"prompt": input_text[i], "raw_prediction": responses[i][0]['generated_text'], "raw_answers": self.dataset[i]['raw_answers']} for i in range(len(responses))]

        return output

    def eval_metric(self, results):
        scores = []
        for sample in results:
            raw_prediction, raw_answers = sample["raw_prediction"], sample["raw_answers"]
            prediction, answers = self.post_process(raw_prediction, raw_answers)
            score = self._metrics(prediction, answers[0])
            scores.append(score)
        return scores
    
    def post_process(self, raw_prediction, ground_truths):
        pred = raw_prediction.strip()
        if pred == "":
            pred = "None"
        pred.strip(".。")

        ground_truth = ground_truths[0]
        return pred, [ground_truth]
    
    def _metrics(self, prediction, ground_truth):
        raise NotImplementedError

    def evaluate(self, model, tokenizer):
        print("Running inference on evaluation dataset...")
        results = self.infer(model, tokenizer)
        print("Evaluating results...")
        metrics = self.eval_metric(results)
        print("Evaluation complete. The result is as follows:")
        print(f"Average score: {sum(metrics) / len(metrics)}")
        return results, metrics


class GPT4Evaluator(BaseEvaluator):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        import openai
        self.client = openai.AzureOpenAI(
            api_key=config['openai_api_key'],
            api_version="2024-02-15-preview"
        )

    def query_gpt4(self, text):
        # try for 5 times
        MAX_TRIAL=5
        for i in range(MAX_TRIAL):
            try:
                chat_completion = self.client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Follow the user's instructions carefully. Respond using markdown."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=80
                )
                response_text = chat_completion.choices[0].message.content
                break
            except Exception as e:
                print("ERROR:", e)
                print(f"error in connecting to OpenAI server for {i+1}-th time. try again")
                response_text = ""
                time.sleep(10)
                
        return response_text

    def parse_gpt4(self, response_text):
        score = re.findall(self.pattern, response_text)
        if score:
            score = float(score[0]) / 10
        else:
            score = 0.0
            print("GPT4没有给出合理的分数:", response_text)
        return score

    @property
    def template(self):
        raise NotImplementedError
    
    @property
    def pattern(self):
        raise NotImplementedError

    def _metrics(self, prediction, ground_truth):
        text = self.template.format(prediction=prediction, ground_truth=ground_truth)
        response_text = self.query_gpt4(text)
        score = self.parse_gpt4(response_text)
        return score
    


class IntentEvaluator(BaseEvaluator):

    def post_process(self, raw_prediction, ground_truths):
        pred = raw_prediction.strip()
        if pred == "":
            pred = "None"
        pred = pred.strip('.。')

        if "```json" in pred:
            try:
                pred = pred[pred.index("```json") + 7:]
                pred = pred[:pred.index("```")]
            except:
                print("unable to parse answer", pred)
                pred = "{}"

        if "\n" in pred:
            pred = [i for i in pred.split("\n") if i][0]

        pred = pred.strip('.。')

        ground_truth = ground_truths[0]
        return pred, [ground_truth]

    def _metrics(self, prediction, ground_truth):
        ground_truth = json.loads(ground_truth)
        try:
            prediction = json.loads(prediction) 
        except:
            print(f"unable to parse prediction {prediction} of example with gt {ground_truth}")
            return 0.0

        intent_em = prediction.get('intent', '') == ground_truth.get('intent', '')

        gt_slots = {(k, str(tuple(sorted([str(i) for i in v]))) if isinstance(v, list) else v) for k, v in ground_truth.get('slots', {}).items()}
        try:
            pred_slots = {(k, str(tuple(sorted([str(i).replace(" ", "") for i in v]))) if isinstance(v, list) else v.replace(" ", "")) for k, v in prediction.get('slots', {}).items()}
        except:
            print(f"OK to parse prediction slots {prediction} of example with gt {ground_truth}, but failed in processing the contents.")
            return 0.0  
                    
        correct_slots = pred_slots.intersection(gt_slots)
        slots_em = (len(correct_slots) == len(pred_slots)) and (len(correct_slots) == len(gt_slots))
        
        return int(intent_em and slots_em)



SummaryTemplate = """
请你进行以下电话总结内容的评分。请依据以下标准综合考量，以确定预测答案与标准答案之间的一致性程度。满分为10分，根据预测答案的准确性、完整性和相关性来逐项扣分。请先给每一项打分并给出总分，再给出打分理由。总分为10分减去每一项扣除分数之和，最低可扣到0分。请以“内容准确性扣x分，详细程度/完整性扣x分，...，总分是：x分"为开头。

1. **内容准确性**：
   - 预测答案是否准确反映了客户问题或投诉的核心要点。
   - 是否有任何关键信息被错误陈述或误解。

2. **详细程度/完整性**：
   - 预测答案中包含的细节是否充分，能否覆盖标准答案中所有重要点。
   - 对于任何遗漏的关键信息，应相应减分。

3. **内容冗余度**：
   - 预测答案是否简洁明了，和标准答案风格一致，不存在冗余信息。
   - 如果预测答案过长或与标准答案风格不一致，需相应减分。

4. **行动指令正确性**：
   - 预测答案对后续处理的建议或请求是否与标准答案相符。
   - 如果处理建议发生改变或丢失，需相应减分。

预测答案：{prediction}
参考答案：{ground_truth}
"""


class SummaryEvaluator(GPT4Evaluator):

    @property
    def pattern(self):
        return r"总分是：(\d+\.\d+|\d+)分"
    
    @property
    def template(self):
        return SummaryTemplate
    

LawTemplate = """
请你进行以下法案判决预测内容的评分。请依据以下标准综合考量，以确定预测答案与标准答案之间的一致性程度。满分为10分，根据预测答案的准确性、完整性和相关性来逐项扣分。请先给每一项打分并给出总分，再给出打分理由。总分为10分减去每一项扣除分数之和，最低可扣到0分。请以“相关性扣x分，完整性扣x分，...，总分是：x分"为开头。

1. **相关性**：预测答案与标准答案的相关程度是最重要的评分标准。如果预测的判决情况与标准答案完全一致，即所有事实和结果都被精确复制或以不同但等效的方式表述，则应给予高分。若只有部分一致或存在偏差，则根据一致的程度适当扣分。如果没有预测判决内容，扣10分。

2. **完整性**：评估预测答案是否涵盖了所有标准答案中提到的关键点，包括但不限于当事人、具体金额、责任判定、费用承担等。如果遗漏重要信息，则应相应扣分。

3. **准确性**：检查预测答案中提及的细节、数字、日期和法律依据是否与标准答案保持一致。任何错误信息均需扣分，并且严重错误应该导致更多的扣分。

4. **客观性与专业性**：预测答案应客观反映法案内容并使用恰当的法律术语。主观臆断或非专业表达需酌情扣分。
   
预测答案：{prediction}
参考答案：{ground_truth}
"""

class LawEvaluator(GPT4Evaluator):

    @property
    def pattern(self):
        return r"总分是：(\d+\.\d+|\d+)分"
    
    @property
    def template(self):
        return LawTemplate


TranslationTemplate = """
You are an expert master in machine translation. Please score the predicted answer against the standard answer out of 10 points based on the following criteria:

Content accuracy: Does the predicted answer accurately reflect the key points of the reference answer?
Level of detail/completeness: Does the predicted answer cover all important points from the standard answer?
Content redundancy: Is the predicted answer concise and consistent with the style of the standard answer?

Respond following the format:"Content accuracy x points, level of detail/completeness x points, ..., total score: x points". The total score is the average of all the scores. Do not give reasons for your scores.

Predicted answer: {prediction}

Reference answer: {ground_truth}

"""

class TranslationEvaluator(GPT4Evaluator):

    @property
    def pattern(self):
        return r"score: *?(\d+\.\d+|\d+) *?point"
    
    @property
    def template(self):
        return TranslationTemplate

    def post_process(self, raw_prediction, ground_truths):
        pred = raw_prediction.strip().split("\n\n")[0]
        if pred == "":
            pred = "None"
        pred.strip(".。")

        ground_truth = ground_truths[0]
        return pred, [ground_truth]
