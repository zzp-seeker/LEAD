
import json
import os
from typing import Dict, List, Optional

import torch


def load_skill_mapping(skills_dir: str) -> dict:
    mapping_path = os.path.join(skills_dir, "skill_mapping.json")
    with open(mapping_path, "r") as f:
        return json.load(f)


def load_skill_content(skills_dir: str, skill_mapping: dict) -> Dict[str, str]:
    contents = {}
    for skill_name, filename in skill_mapping["skill_files"].items():
        filepath = os.path.join(skills_dir, filename)
        with open(filepath, "r") as f:
            contents[skill_name] = f.read().strip()
    return contents


class SkillProvider:

    def __init__(self, skills_dir: str, skill_all: bool = False):
        self.skills_dir = skills_dir
        self.skill_all = skill_all
        self.skill_mapping = load_skill_mapping(skills_dir)
        self.skill_contents = load_skill_content(skills_dir, self.skill_mapping)
        self.task_to_skill = self.skill_mapping["task_to_skill"]
        self.task_keywords: Dict[str, List[str]] = self.skill_mapping.get("task_keywords", {})

        if self.skill_all:
            self._all_skills_text = self._build_all_skills_text()

    def _build_all_skills_text(self) -> str:
        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        for skill_name, content in self.skill_contents.items():
            if skill_name != "general_skills":
                parts.append(content)
        return "\n\n".join(parts)

    def _get_skill_text(self, task_type: Optional[str]) -> str:
        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        if task_type:
            mapped_name = self.task_to_skill.get(task_type)
            if mapped_name and mapped_name in self.skill_contents:
                parts.append(self.skill_contents[mapped_name])
        return "\n\n".join(parts)

    def get_privileged_info(self, gamefile: str) -> str:
        if self.skill_all:
            return self._all_skills_text
        matched_task = None
        for task_type in self.task_to_skill:
            if task_type in gamefile:
                matched_task = task_type
                break
        return self._get_skill_text(matched_task)

    def get_privileged_info_from_prompt(self, prompt_text: str) -> str:
        if self.skill_all:
            return self._all_skills_text
        text_lower = prompt_text.lower()
        matched_tasks = []
        for task_type, keywords in self.task_keywords.items():
            if keywords and any(kw in text_lower for kw in keywords):
                matched_tasks.append(task_type)

        if not matched_tasks:
            return self._get_skill_text(None)

        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        for task_type in matched_tasks:
            mapped_name = self.task_to_skill.get(task_type)
            if mapped_name and mapped_name in self.skill_contents:
                parts.append(self.skill_contents[mapped_name])
        return "\n\n".join(parts)

    def get_privileged_info_from_data_source(self, data_source: str, prompt_text: str) -> str:
        if self.skill_all:
            return self._all_skills_text

        task_type = None
        if data_source == "popqa":
            task_type = "entity_attribute_lookup"
        elif data_source in ("nq", "triviaqa"):
            task_type = "direct_retrieval"
        elif data_source == "hotpotqa":
            task_type = "multi_hop_reasoning"
        else:
            text_lower = prompt_text.lower()
            if "which" in text_lower and "or" in text_lower and "for" not in text_lower:
                task_type = "compare"
            elif data_source in ("2wikimultihopqa", "musique", "bamboogle"):
                task_type = "multi_hop_reasoning"

        if task_type is None and data_source == "text":
            return self.get_privileged_info_from_prompt(prompt_text)
        return self._get_skill_text(task_type)


def compute_rlsd_token_advantage(
    seq_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    rlsd_lambda: float = 0.5,
    rlsd_clip_eps: float = 0.2,
    cut_negative: bool = False,
) -> torch.Tensor:
    with torch.no_grad():
        first_valid = response_mask.int().argmax(dim=-1)
        batch_indices = torch.arange(seq_advantages.size(0), device=seq_advantages.device)
        A_seq = seq_advantages[batch_indices, first_valid]

        delta_t = teacher_log_probs - student_log_probs

        sign_A = torch.sign(A_seq).unsqueeze(-1)
        w_t = torch.exp(sign_A * delta_t)
        w_lower = 1.0 if cut_negative else (1.0 - rlsd_clip_eps)
        w_t = torch.clamp(w_t, w_lower, 1.0 + rlsd_clip_eps)

        A_seq_expanded = A_seq.unsqueeze(-1)
        token_advantages = A_seq_expanded * ((1.0 - rlsd_lambda) + rlsd_lambda * w_t)
        token_advantages = token_advantages * response_mask

    return token_advantages
