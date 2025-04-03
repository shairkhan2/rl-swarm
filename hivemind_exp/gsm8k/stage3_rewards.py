import os
import random
import re
from difflib import SequenceMatcher
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
    except (IndexError, ValueError, AttributeError) as e:
        # Log or handle exception if needed
        return default


def extract_xml_identity(text: str) -> str:
    return safe_extract("<majority>", text, "<majority>", "</majority>")


def extract_xml_final_answer(text: str) -> str:
    return safe_extract("<answer>", text, "<answer>", "</answer>")


def extract_xml_question(text: str) -> str:
    return safe_extract("<question>", text, "<question>", "</question>")


def extract_xml_ids(text: str) -> list:
    """
    Extract multiple <student>...</student> occurrences robustly.
    """
    ids = []
    try:
        parts = text.split("<student>")[1:]  # everything after the first <student>
        for p in parts:
            # each p is "SomeID</student>..."
            id_chunk = p.split("</student>", maxsplit=1)[0].strip()
            ids.append(id_chunk)
    except Exception as e:
        # Handle/log if needed
        pass
    return ids


def extract_xml_choices(text: str) -> list:
    """
    Extract multiple <identify>...</identify> occurrences robustly.
    """
    choices = []
    try:
        parts = text.split("<identify>")[1:]
        for p in parts:
            choice_chunk = p.split("</identify>", maxsplit=1)[0].strip()
            choices.append(choice_chunk)
    except Exception as e:
        # Handle/log if needed
        pass
    return choices


def extract_original_question(text: str) -> str:
    """
    Attempt to parse out the original question from known substrings.
    Falls back to an empty string if not found.
    """
    try:
        # first split
        first_split = text.split("  \n\nThe following answers to this question were suggested:", maxsplit=1)
        before_feedback = first_split[0] if len(first_split) == 2 else ""
        # second split
        second_split = before_feedback.split("The question we were given is: ", maxsplit=1)
        question_part = second_split[-1].strip() if len(second_split) > 0 else ""
        return question_part
    except Exception as e:
        # fallback
        return ""


def extract_answers(text: str) -> dict:
    """
    Return {id: answer_text} dict, with extra checks to avoid index errors.
    """
    answers = {}
    try:
        # split off the part before feedback
        first_split = text.split(
            "  \nAfter comparing these answers, the following feedback was given about which answer is best: \n",
            maxsplit=1
        )
        # if no feedback delimiter is found, just use the entire text for splitting
        relevant_text = first_split[0] if len(first_split) > 0 else text

        # now parse each <student> block
        raw_blocks = relevant_text.split("<student>")[1:]
        for block in raw_blocks:
            parts = block.split("</student>", maxsplit=1)
            if len(parts) < 2:
                continue
            student_id = parts[0].strip()
            # remainder might be " said \n<whatever>"
            remainder = parts[1]
            # find the " said \n" substring and get what's after
            said_split = remainder.split(" said \n", maxsplit=1)
            if len(said_split) == 2:
                ans_text = said_split[1].strip()
                answers[student_id] = ans_text
    except Exception as e:
        # Could log the exception or set answers = {}
        pass
    return answers


def count_xml(text: str) -> float:
    """
    Partial scoring for presence of final-stage tags,
    with some penalty for extra text after closing tags.
    """
    try:
        count = 0.0
        if text.count("<summarize_feedback>\n") == 1:
            count += 0.125
        if text.count("\n</summarize_feedback>\n") == 1:
            count += 0.125
        if text.count("<majority>\n") == 1:
            count += 0.125
        if text.count("\n</majority>\n") == 1:
            count += 0.125
        if text.count("<question>\n") == 1:
            count += 0.125
        if text.count("\n</question>\n") == 1:
            count += 0.125
        if text.count("<think>\n") == 1:
            count += 0.125
        if text.count("\n</think>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            # penalty for trailing text
            tail_after_answer = text.split("\n</answer>\n")[-1]
            count -= len(tail_after_answer) * 0.001 if tail_after_answer else 0
        if text.count("\n</answer>") == 1:
            count += 0.125
            tail2 = text.split("\n</answer>")[-1]
            count -= (len(tail2) - 1) * 0.001 if len(tail2) > 1 else 0
        return count
    except Exception as e:
        return 0.0


def swarm_majority(choices):
    """
    Return list of items that appear most frequently in 'choices'.
    """
    try:
        votes = {}
        max_votes = 0
        for c in choices:
            votes[c] = votes.get(c, 0) + 1
            if votes[c] > max_votes:
                max_votes = votes[c]
        majority = [k for k, v in votes.items() if v >= max_votes and k.strip()]
        return majority
    except Exception:
        return []


# ======================= REWARD FUNCTIONS =======================


def consensus_reward_func(
    prompts, completions, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    # Wrap everything in try/except so you don't crash the worker
    try:
        responses = [completion[0]["content"] for completion in completions]
        p = prompts[0][-1]["content"]
        critic_choices = extract_xml_choices(p)
        majority_choices = swarm_majority(critic_choices)
        extracted_responses = [extract_xml_identity(r) for r in responses]

        # Logging snippet
        if (random.random() < 0.01) and logging:
            try:
                os.makedirs(
                    f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                    exist_ok=True,
                )
                log_file = os.path.join(
                    "model_output_samples",
                    f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                    "consensus_samps.txt",
                )
                with open(log_file, "a") as f:
                    f.write("-" * 20)
                    out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nCritic Choice Distribution:\n{critic_choices}\n\nExtracted:\n{extracted_responses[0]}\n\nGot reward? {extracted_responses[0] in majority_choices}"
                    f.write(out_line)
            except Exception:
                pass

        return [1.0 * weighting if r in majority_choices else 0.0 for r in extracted_responses]
    except Exception as e:
        # If something goes wrong, return zeros
        return [0.0 for _ in completions]


def question_recreation_reward_func(
    prompts, completions, weighting=1.0, logging=False, **kwargs
) -> list[float]:
    try:
        responses = [completion[0]["content"] for completion in completions]
        p = prompts[0][-1]["content"]
        q = extract_original_question(p)
        recreated_qs = [extract_xml_question(r) for r in responses]

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
                    "question_recreation_samps.txt",
                )
                with open(log_file, "a") as f:
                    f.write("-" * 20)
                    out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nOriginal Question:\n{q}\n\nExtracted recreation:\n{recreated_qs[0]}\n\nSimilarity: {SequenceMatcher(None, recreated_qs[0], q).ratio()}"
                    f.write(out_line)
            except Exception:
                pass

        return [SequenceMatcher(None, r, q).ratio() * weighting for r in recreated_qs]
    except Exception as e:
        return [0.0 for _ in completions]


def concensus_correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    try:
        responses = [completion[0]["content"] for completion in completions]
        p = prompts[0][-1]["content"]
        agent_answers = extract_answers(p)
        extracted_responses = [extract_xml_identity(r) for r in responses]
        chosen_rewards = []

        for r in extracted_responses:
            cur_reward = 0
            if r in agent_answers:
                try:
                    # Check correctness with stage1 logic
                    if stage1_rewards.extract_xml_answer(agent_answers[r]) == answer[0]:
                        cur_reward += 1.0
                    if stage1_rewards.extract_xml_answer(agent_answers[r]).isdigit():
                        cur_reward += 0.5

                    # Format checks
                    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
                    if re.match(pattern, agent_answers[r]):
                        cur_reward += 0.5
                    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
                    if re.match(pattern, agent_answers[r]):
                        cur_reward += 0.5

                    # Additional partial scoring from stage1
                    cur_reward += stage1_rewards.count_xml(agent_answers[r])

                except Exception:
                    pass
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
                # Check if all agent submissions are indeed incorrect
                try:
                    agent_as = [
                        stage1_rewards.extract_xml_answer(agent_answers[id])
                        for id in agent_answers
                    ]
                    check_submissions = [
                        True if r == a else False for r, a in zip(agent_as, answer)
                    ]
                    if all(check_submissions):
                        cur_reward += 10
                except Exception:
                    pass

            chosen_rewards.append(cur_reward)

        # Optional logging
        if (random.random() < 0.01) and logging and len(extracted_responses) > 0:
            first_id = extracted_responses[0]
            if first_id in agent_answers:
                try:
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
                            f"Extracted:\n{agent_answers[first_id]}\n\n"
                            f"Reward for choice: {chosen_rewards[0]}"
                        )
                        f.write(out_line)
                except Exception:
                    pass

        return [r * weighting for r in chosen_rewards]

    except Exception as e:
        return [0.0 for _ in completions]


def final_correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=False, **kwargs
) -> list[float]:
    try:
        responses = [completion[0]["content"] for completion in completions]
        p = prompts[0][-1]["content"]
        extracted_responses = [extract_xml_final_answer(r) for r in responses]

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
                    "final_answer_correctness_samples.txt",
                )
                with open(log_file, "a") as f:
                    f.write("-" * 20)
                    out_line = (
                        f"Prompt:\n{p}\n\n"
                        f"Answer:\n{answer[0]}\n\n"
                        f"Response:\n{responses[0]}\n\n"
                        f"Extracted:\n{extracted_responses[0]}"
                    )
                    f.write(out_line)
            except Exception:
                pass

        return [
            1.0 * weighting if r == a else 0.0 for r, a in zip(extracted_responses, answer)
        ]
    except Exception as e:
        return [0.0 for _ in completions]


def strict_format_reward_func(
    completions, weighting=0.5, logging=False, **kwargs
) -> list[float]:
    try:
        pattern = (
            r"^<summarize_feedback>\n.*?\n</summarize_feedback>\n"
            r"<majority>\n.*?\n</majority>\n"
            r"<question>\n.*?\n</question>\n"
            r"<think>\n.*?\n</think>\n"
            r"<answer>\n.*?\n</answer>\n$"
        )
        responses = [completion[0]["content"] for completion in completions]
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
                    "s3_strict_format_samps.txt",
                )
                with open(log_file, "a") as f:
                    f.write("-" * 20)
                    out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
                    f.write(out_line)
            except Exception:
                pass

        return [1.0 * weighting if match else 0.0 for match in matches]
    except Exception as e:
        return [0.0 for _ in completions]


def soft_format_reward_func(
    completions, weighting=0.5, logging=False, **kwargs
) -> list[float]:
    try:
        pattern = (
            r"<summarize_feedback>.*?</summarize_feedback>\s*"
            r"<majority>.*?</majority>\s*"
            r"<question>.*?</question>\s*"
            r"<think>.*?</think>\s*"
            r"<answer>.*?</answer>"
        )
        responses = [completion[0]["content"] for completion in completions]
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
                    "s3_soft_format_samps.txt",
                )
                with open(log_file, "a") as f:
                    f.write("-" * 20)
                    out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
                    f.write(out_line)
            except Exception:
                pass

        return [1.0 * weighting if match else 0.0 for match in matches]
    except Exception as e:
        return [0.0 for _ in completions]


def xmlcount_reward_func(
    completions, weighting=1.0, logging=False, **kwargs
) -> list[float]:
    try:
        contents = [completion[0]["content"] for completion in completions]
        if (random.random() < 0.01) and logging and len(contents) > 0:
            try:
                os.makedirs(
                    f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                    exist_ok=True,
                )
                log_file = os.path.join(
                    "model_output_samples",
                    f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                    "count_xml_samps.txt",
                )
                with open(log_file, "a") as f:
                    f.write("-" * 20)
                    out_line = f"\nResponse:\n{contents[0]}\n\nCount reward: {count_xml(contents[0])}"
                    f.write(out_line)
            except Exception:
                pass

        return [count_xml(c) * weighting for c in contents]
    except Exception as e:
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
    Dummy reward function that accumulates all rewards into one + saves JSON to node.outputs
    Includes error handling so it won't crash the entire process if an exception is thrown.
    """
    try:
        # Step 1: gather partial rewards
        consensus_reward = consensus_reward_func(prompts, completions, logging=logging)
        concensus_correctness = concensus_correctness_reward_func(
            prompts, completions, answer, logging=logging
        )
        question_recreation_reward = question_recreation_reward_func(
            prompts, completions, logging=logging
        )
        final_correctness = final_correctness_reward_func(
            prompts, completions, answer, logging=logging
        )
        strict_format_reward = strict_format_reward_func(completions, logging=logging)
        soft_format_reward = soft_format_reward_func(completions, logging=logging)
        xmlcount_reward = xmlcount_reward_func(completions, logging=logging)

        total_reward = [
            sum(tup)
            for tup in zip(
                consensus_reward,
                concensus_correctness,
                question_recreation_reward,
                final_correctness,
                strict_format_reward,
                soft_format_reward,
                xmlcount_reward,
            )
        ]

        # Step 2: store in node outputs
        prompt = prompts[0][-1]["content"]
        question = extract_original_question(prompt)

        if output_signal_selector == "max":
            maximal_reward_idx = int(np.argmax(total_reward)) if len(total_reward) else 0
            responses = [completion[0]["content"] for completion in completions]
            chosen_response = responses[maximal_reward_idx] if responses else ""

            output_data = {
                "question": question,
                "answer": answer[0] if len(answer) > 0 else "",
                "stage3_prompt": prompt,
                "final_agent_decision": {node.key: chosen_response},
            }
        else:
            output_data = {}

        if output_signal_selector is not None:
            node.outputs = output_data
            node.rewards = total_reward

        # Hivemind reward function should return list[float] of same length as completions,
        # but we return 0.0 so it won't overshadow partial rewards.
        return [0.0 for _ in total_reward]

    except Exception as e:
        # If something catastrophic happens, log or handle
        # Return zeros to avoid killing the child process
        return [0.0 for _ in completions]
