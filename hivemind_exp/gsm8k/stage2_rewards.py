import os
import random
import re
import numpy as np

import hivemind_exp.gsm8k.stage1_rewards as stage1_rewards
from hivemind_exp.hivemind_utils import HivemindNode


def safe_extract(substring, text, start_tag, end_tag, default=""):
    """
    Safely extract text between start_tag and end_tag in 'text'.
    If not found or indices are out of range, return default.
    """
    try:
        split_start = text.split(start_tag, maxsplit=1)
        if len(split_start) < 2:
            return default
        remainder = split_start[1]
        split_end = remainder.split(end_tag, maxsplit=1)
        if len(split_end) < 2:
            return default
        return split_end[0].strip()
    except Exception:
        return default


def extract_xml_identity(text: str) -> str:
    """
    Extract text between <identify> and </identify> safely.
    """
    return safe_extract("<identify>", text, "<identify>", "</identify>")


def extract_xml_ids(text: str) -> list:
    """
    Extract multiple IDs from <student>some_id</student> blocks.
    """
    ids = []
    try:
        parts = text.split("<student>")[1:]
        for p in parts:
            chunk = p.split("</student>", maxsplit=1)[0].strip()
            ids.append(chunk)
    except Exception:
        pass
    return ids


def extract_original_question(text: str) -> str:
    """
    Attempt to parse out the original question from known substrings.
    Falls back to an empty string if not found.
    """
    try:
        first_split = text.split("  \n\nThe following answers to this question were suggested:", maxsplit=1)
        before_answers = first_split[0] if len(first_split) == 2 else ""
        second_split = before_answers.split("The question we were given is: ", maxsplit=1)
        question_part = second_split[-1].strip() if second_split else ""
        return question_part
    except Exception:
        return ""


def extract_answers(text: str) -> dict:
    """
    Return {id: answer_text}, handling potential missing tags gracefully.
    """
    answers = {}
    try:
        blocks = text.split("<student>")[1:]
        for b in blocks:
            parts = b.split("</student>", maxsplit=1)
            if len(parts) < 2:
                continue
            student_id = parts[0].strip()

            remainder = parts[1].split(" said \n", maxsplit=1)
            if len(remainder) < 2:
                continue
            ans_text = remainder[1].strip()
            answers[student_id] = ans_text
    except Exception:
        pass
    return answers


def count_xml(text: str) -> float:
    """
    Count partial points for each recognized tag, with minor penalty if extra text appears.
    """
    try:
        count = 0.0
        # compare tags
        if text.count("<compare>\n") == 1:
            count += 0.125
        if text.count("\n</compare>\n") == 1:
            count += 0.125

        # explain tags
        if text.count("<explain>\n") == 1:
            count += 0.125
        if text.count("\n</explain>\n") == 1:
            count += 0.125

        # identify tags, also penalize trailing text
        if text.count("\n<identify>\n") == 1:
            count += 0.125
            tail_1 = text.split("\n</identify>\n")[-1]
            count -= len(tail_1) * 0.001 if tail_1 else 0
        if text.count("\n</identify>") == 1:
            count += 0.125
            tail_2 = text.split("\n</identify>")[-1]
            count -= (len(tail_2) - 1) * 0.001 if len(tail_2) > 1 else 0

        return count
    except Exception:
        return 0.0


# ==================== REWARD FUNCTIONS ====================

def proper_id_reward_func(
    prompts, completions, answer, weighting=2.0, logging=True, **kwargs
) -> list[float]:
    """
    Reward if the chosen <identify>... tag matches any of the valid <student>... IDs from the prompt.
    """
    try:
        responses = [c[0]["content"] for c in completions]
        p = prompts[0][-1]["content"]
        agent_ids = extract_xml_ids(p)
        extracted_responses = [extract_xml_identity(r) for r in responses]

        # Optional logging
        if (random.random() < 0.01) and logging and len(responses) > 0:
            try:
                os.makedirs(
                    f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                    exist_ok=True,
                )
                log_file = os.path.join(
                    "model_output_samples",
                    f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                    "id_extact_samps.txt",
                )
                with open(log_file, "a") as f:
                    f.write("-" * 20)
                    out_line = (
                        f"\nPrompt:\n{p}\n\n"
                        f"Response:\n{responses[0]}\n\n"
                        f"Valid IDs:\n{agent_ids}\n\n"
                        f"Extracted:\n{extracted_responses[0]}\n\n"
                        f"Got reward? {extracted_responses[0] in agent_ids}"
                    )
                    f.write(out_line)
            except Exception:
                pass

        return [weighting if r in agent_ids else 0.0 for r in extracted_responses]
    except Exception:
        return [0.0 for _ in completions]


def correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=True, **kwargs
) -> list[float]:
    """
    Reward if the model identifies a student whose answer is correct or "None" if all are wrong.
    """
    chosen_rewards = []
    try:
        responses = [c[0]["content"] for c in completions]
        p = prompts[0][-1]["content"]
        agent_answers = extract_answers(p)
        extracted_responses = [extract_xml_identity(r) for r in responses]

        for r in extracted_responses:
            cur_reward = 0.0
            try:
                if r in agent_answers:
                    # Compare to ground-truth
                    if stage1_rewards.extract_xml_answer(agent_answers[r]) == answer[0]:
                        cur_reward += 1.0
                    if stage1_rewards.extract_xml_answer(agent_answers[r]).isdigit():
                        cur_reward += 0.5

                    # Format checks from stage1
                    pat_strict = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
                    if re.match(pat_strict, agent_answers[r]):
                        cur_reward += 0.5

                    pat_soft = r"<think>.*?</think>\s*<answer>.*?</answer>"
                    if re.match(pat_soft, agent_answers[r]):
                        cur_reward += 0.5

                    # Additional partial scoring from stage1
                    cur_reward += stage1_rewards.count_xml(agent_answers[r])

                elif r in [
                    "None",
                    "No one",
                    "All answers are wrong",
                    "All answers were wrong",
                    "All are wrong",
                    "All were wrong",
                    "None are correct",
                    "None were correct",
                    "No one is correct",
                ]:
                    # If the model says "None" are correct, check if indeed all are wrong
                    try:
                        agent_as = [
                            stage1_rewards.extract_xml_answer(agent_answers[sid])
                            for sid in agent_answers
                        ]
                        check_submissions = [ra == a for ra, a in zip(agent_as, answer)]
                        if all(check_submissions):
                            # Big reward for noticing all are incorrect
                            cur_reward += 10
                    except Exception:
                        pass
            except Exception:
                pass

            chosen_rewards.append(cur_reward)

        # Logging
        if (random.random() < 0.01) and logging and len(responses) > 0:
            try:
                first_id = extracted_responses[0]
                if first_id in agent_answers:
                    os.makedirs(
                        f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                        exist_ok=True,
                    )
                    log_file = os.path.join(
                        "model_output_samples",
                        f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                        "correctness_samps.txt",
                    )
                    with open(log_file, "a") as f:
                        f.write("-" * 20)
                        out_line = (
                            f"\nPrompt:\n{p}\n\n"
                            f"Response:\n{responses[0]}\n\n"
                            f"Chosen answer ID:\n{first_id}\n\n"
                            f"Extracted:\n{agent_answers.get(first_id, '')}\n\n"
                            f"Reward for choice: {chosen_rewards[0]}"
                        )
                        f.write(out_line)
            except Exception:
                pass

        return [r * weighting for r in chosen_rewards]
    except Exception:
        # if something fails outright, return zeros
        return [0.0 for _ in completions]


def strict_format_reward_func(
    completions, weighting=0.5, logging=True, **kwargs
) -> list[float]:
    """
    Reward if the completion matches a strict multi-line format with <compare>, <explain>, <identify>.
    """
    try:
        pattern = (
            r"^<compare>\n.*?\n</compare>\n"
            r"<explain>\n.*?\n</explain>\n"
            r"<identify>\n.*?\n</identify>\n$"
        )
        responses = [c[0]["content"] for c in completions]
        matches = [re.match(pattern, r) for r in responses]

        # Logging
        if (random.random() < 0.01) and logging and len(responses) > 0:
            try:
                os.makedirs(
                    f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                    exist_ok=True,
                )
                log_file = os.path.join(
                    "model_output_samples",
                    f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                    "s2_strict_format_samps.txt",
                )
                with open(log_file, "a") as f:
                    f.write("-" * 20)
                    out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
                    f.write(out_line)
            except Exception:
                pass

        return [1.0 * weighting if m else 0.0 for m in matches]
    except Exception:
        return [0.0 for _ in completions]


def soft_format_reward_func(
    completions, weighting=0.5, logging=True, **kwargs
) -> list[float]:
    """
    Reward if the completion has <compare>.*?</compare>, <explain>.*?</explain>, <identify>.*?</identify> in any order/whitespace.
    """
    try:
        pattern = r"<compare>.*?</compare>\s*<explain>.*?</explain>\s*<identify>.*?</identify>"
        responses = [c[0]["content"] for c in completions]
        matches = [re.match(pattern, r) for r in responses]

        # Logging
        if (random.random() < 0.01) and logging and len(responses) > 0:
            try:
                os.makedirs(
                    f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                    exist_ok=True,
                )
                log_file = os.path.join(
                    "model_output_samples",
                    f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                    "s2_soft_format_samps.txt",
                )
                with open(log_file, "a") as f:
                    f.write("-" * 20)
                    out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
                    f.write(out_line)
            except Exception:
                pass

        return [1.0 * weighting if m else 0.0 for m in matches]
    except Exception:
        return [0.0 for _ in completions]


def xmlcount_reward_func(
    completions, weighting=1.0, logging=True, **kwargs
) -> list[float]:
    """
    Reward partial points for each recognized Stage-2 XML tag.
    """
    try:
        contents = [c[0]["content"] for c in completions]

        # Logging
        if (random.random() < 0.01) and logging and len(contents) > 0:
            try:
                os.makedirs(
                    f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                    exist_ok=True,
                )
                log_file = os.path.join(
                    "model_output_samples",
                    f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                    "strict_format_samps.txt",
                )
                with open(log_file, "a") as f:
                    f.write("-" * 20)
                    out_line = f"\nResponse:\n{contents[0]}\n\nCount reward: {count_xml(contents[0])}"
                    f.write(out_line)
            except Exception:
                pass

        return [count_xml(c) * weighting for c in contents]
    except Exception:
        return [0.0 for _ in completions]


def top_k_cumulative_reward(
    prompts,
    completions,
    answer,
    logging=False,
    **kwargs,
) -> list[float]:
    """
    Accumulate multiple reward signals and return a total for top_k selection.
    """
    try:
        proper_id_reward = proper_id_reward_func(prompts, completions, answer, logging=logging)
        correctness = correctness_reward_func(prompts, completions, answer, logging=logging)
        strict_format = strict_format_reward_func(completions, logging=logging)
        soft_format = soft_format_reward_func(completions, logging=logging)
        xmlcount = xmlcount_reward_func(completions, logging=logging)

        total_reward = [
            sum(values)
            for values in zip(proper_id_reward, correctness, strict_format, soft_format, xmlcount)
        ]
        return total_reward
    except Exception:
        # Return zero if anything goes wrong
        return [0.0 for _ in completions]


def hivemind_cumulative_reward(
    node: HivemindNode,
    prompts,
    completions,
    answer,
    logging=False,
    output_signal_selector="max",
    **kwargs,
) -> list[float]:
    """
    Accumulates all rewards, optionally chooses the max, and stores data in node.outputs.
    """
    try:
        proper_id_reward = proper_id_reward_func(prompts, completions, answer, logging=logging)
        correctness = correctness_reward_func(prompts, completions, answer, logging=logging)
        strict_format = strict_format_reward_func(completions, logging=logging)
        soft_format = soft_format_reward_func(completions, logging=logging)
        xmlcount = xmlcount_reward_func(completions, logging=logging)

        total_reward = [
            sum(vals)
            for vals in zip(proper_id_reward, correctness, strict_format, soft_format, xmlcount)
        ]

        question = extract_original_question(prompts[0][-1]["content"]) if prompts else ""

        if output_signal_selector == "max":
            try:
                idx = int(np.argmax(total_reward)) if len(total_reward) else 0
                responses = [c[0]["content"] for c in completions]
                chosen = responses[idx] if idx < len(responses) else ""
                output_data = {
                    "question": question,
                    "answer": answer[0] if len(answer) > 0 else "",
                    "stage2_prompt": prompts[0][-1]["content"] if prompts else "",
                    "agent_opinion": {node.key: chosen},
                }
            except Exception:
                output_data = {}
        else:
            output_data = {}

        if output_signal_selector is not None:
            node.outputs = output_data
            node.rewards = total_reward

        # Return a list of zeros to indicate we don't overshadow partial training steps
        return [0.0 for _ in total_reward]

    except Exception:
        return [0.0 for _ in completions]
