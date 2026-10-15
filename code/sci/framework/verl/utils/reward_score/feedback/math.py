# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from EleutherAI/lm-evaluation-harness
import re
import signal
from typing import Optional
from math_verify import parse as mv_parse, verify as mv_verify

FORMAT_PENALTY = False


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else ""


def remove_boxed(s: str) -> str:
    left = r"\boxed{"
    if s[: len(left)] == left and  s[-1] == "}":
        return s[len(left) : -1]
    else:
        return ""


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return extracted_pred == gt, extracted_pred


def verify(
    solution_str: str, answer: str, pause_tokens_index: Optional[list[int]] = None
) -> bool:
    correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
    if pred is None:
        pred = ""

    if not correct and pred != "":
        try:
            with timeout(seconds=5):
                gold_expr = mv_parse(answer)
                pred_expr = mv_parse(pred)
                correct = mv_verify(gold_expr, pred_expr)
        except Exception:
            pass
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info = None,
    pause_tokens_index: Optional[list[int]] = None,
    format_feedback: bool = True,
    correctness_feedback: bool = False,
) -> float:
    split = extra_info.get("split", "test")
    was_truncated = extra_info.get("truncated", False)

    correct, pred = verify(solution_str, ground_truth, pause_tokens_index)

    reward = 1.0 if correct else 0.0
    score = reward
    incorrect_format = pred is None or pred == ""
    was_truncated = extra_info.get("truncated", False)
    if FORMAT_PENALTY and split == "train" and incorrect_format and (not was_truncated):
        score -= 0.5

    feedback = ""
    if incorrect_format and not was_truncated and format_feedback:
        feedback = "Your answer had the wrong format. The solution must be given in the format: \\boxed{your_answer}."
    elif was_truncated and format_feedback:
        feedback = "Your response was truncated because it exceeded the maximum length."
    elif not correct and correctness_feedback:
        feedback = f"Your answer is incorrect. The correct answer is {ground_truth}."

    return {
        "score": score,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": feedback,
    }
