# Code

Two suites, each with its own framework.

## Scientific reasoning (`sci/`)

`framework/` is the complete framework tree shared by all four methods, taken at
a single commit. `methods/<method>/files/` holds the files that method adds or
modifies, as complete files under their framework-relative paths.

```
MODEL_PATH=/path/to/Qwen3-4B DATA_DIR=../data \
  bash sci/scripts/run_train.sh lead physics qwen3-4b
```

`run_train.sh` copies the selected method's files onto the framework tree before
training. The four methods differ in these files only:

| Method | Files added | Files modified |
| --- | --- | --- |
| grpo | none | 6 |
| sdpo | none | 6 |
| rlsd | `rlsd_utils.py`, `self_future_teacher.py` | 7 |
| lead | `crossfit_teacher.py`, `sfdamp_teacher.py` | 7 |

## Interactive agents (`agent/`)

The complete agent training framework is included under `agent/framework/`, with
the LEAD, RLSD, SDPO, and GRPO changes applied. From the `code/` directory:

```
MODEL_PATH=/path/to/Qwen3-4B AGENT_DATA_DIR=/path/to/agent/data \
  bash agent/scripts/run_train.sh lead webshop qwen3-4b
```

The entry points are `verl.trainer.main_a7` for LEAD, `verl.trainer.main_rlsd`
for RLSD and `verl.trainer.main_sdar` for GRPO and SDPO.

Four of the twenty-four agent runs were interrupted and restarted from an
intermediate checkpoint of the same run: WebShop with LEAD at both scales,
WebShop with GRPO at 8B, and Search with SDPO at 4B. Their configurations carry
`resume_mode: resume_path` and take the checkpoint directory from
`RESUME_PATH`; every other run starts from the base model.

## Configurations

One file per reported run, `<method>_<task>_<scale>.yaml`, holding the
configuration as resolved by the framework at the start of that run. See the
top-level README for launch instructions.

## Environment

Every path in a configuration is an environment-variable reference resolved by
OmegaConf at load time (`${oc.env:MODEL_PATH}` and so on). The launchers require
`MODEL_PATH` plus `DATA_DIR` (scientific reasoning) or `AGENT_DATA_DIR`
(agents), and default `CKPT_DIR` to a directory under the working
directory.

Each suite includes its own `framework/requirements.txt` and
`framework/pyproject.toml`. The scientific-reasoning runs used eight H800 GPUs;
agent resource settings vary by scenario and are recorded in their YAML files.
