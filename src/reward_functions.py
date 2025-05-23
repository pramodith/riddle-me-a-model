import random
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_xml_think(text: str) -> str:
    think = text.split("<think>")[-1]
    think = think.split("</think>")[0]
    return think.strip()

def correctness_reward_func(prompts, completions, answer, quality_score, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    if random.random() < 0.05:
        print('-'*20)
        print(f"Question:\n{q}")
        print(f"Correct Answer:\n{answer[0]}")
        print(f"Responses:\n{responses[0]}")
        print(f"Extracted:\n{extracted_responses[0]}")
    
    return [1 + qs if r.strip() != "" and (a.lower() in r.lower() or r.lower() in a.lower()) else 0 for r, a, qs in zip(extracted_responses, answer, quality_score)]

def answer_length_reward_func(prompts, completions, answer, quality_score, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards  = []
    for r, a in zip(extracted_responses, answer):
        rewards.append(0)
        if a in r or r in a:
            if len(r) > len(a):
                rewards[-1] = -0.001 * (len(r) - len(a))
        else:
            if len(r) < len(a):
                rewards[-1] = -0.005 * (len(a) - len(r))
    return rewards


def think_length_reward_func(prompts, completions, answer, quality_score, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_think(r) for r in responses]
    rewards = [0.5 if 100 < len(r) < 1000 else -0.5 for r in extracted_responses]
    return rewards

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>") == 1:
        count += 0.125
    if text.count("</think>") == 1:
        count += 0.125
    if text.count("<answer>") == 1:
        count += 0.125
    if text.count("</answer>") == 1:
        count += 0.125
    return count

def xmlcount_reward_func(prompts, completions, answer, quality_score, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

