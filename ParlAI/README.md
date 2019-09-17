# Human Evaluation
This folder contains the code for evaluating the evidence selected by agents trained in [convince/allennlp](https://github.com/ethanjperez/convince/tree/master/allennlp).
We used [ParlAI](https://parl.ai/) ([March 12, 2019 commit](https://github.com/ethanjperez/ParlAI/tree/baa839db2120c837562620386cbe1167bd0e9109)), making changes specific to our evaluation setup in a few files.

## Overview
This repo reads the evidence selections from evaluation/inference log files of trained evidence agents (available [here](https://github.com/ethanjperez/convince/tree/master/allennlp/eval/mturk)).
We then run launch HITs (human evaluation jobs) on Amazon Mechanical Turk, using ParlAI's MTurk code ([convince/ParlAI/parlai/mturk](https://github.com/ethanjperez/convince/tree/master/ParlAI/parlai/mturk) - no GPU required).
We made our own ParlAI task which contains all code specific to our evaluations ([convince/ParlAI/parlai/mturk/tasks/context_evaluator](https://github.com/ethanjperez/convince/tree/master/ParlAI/parlai/mturk/tasks/context_evaluator)) - we overview the files in this task-specific folder below:

<table>
<tr>
    <td> <b> Python File </b> </td>
    <td> <b> Functionality </b> </td>
</tr>
<tr>
    <td> <a href="https://github.com/ethanjperez/convince/tree/master/ParlAI/parlai/mturk/tasks/context_evaluator/run.py">run.py</a> </td>
    <td> Initialize, launch, and end HITs </td>
</tr>
<tr>
    <td> <a href="https://github.com/ethanjperez/convince/tree/master/ParlAI/parlai/mturk/tasks/context_evaluator/task_configs.py">task_configs.py</a> </td>
    <td> Human evaluation "hyperparameters" </td>
</tr>
<tr>
    <td> <a href="https://github.com/ethanjperez/convince/tree/master/ParlAI/parlai/mturk/tasks/context_evaluator/worlds.py">worlds.py</a> </td>
    <td> Logic for evaluating evidence and saving results. </td>
</tr>
<tr>
    <td> <a href="https://github.com/ethanjperez/convince/tree/master/ParlAI/parlai/mturk/tasks/context_evaluator/worlds_onboard.py">worlds_onboard.py</a> </td>
    <td> Logic for the Onboarding World. Filters out workers based on performance on a few example "easy" evaluations. </td>
</tr>
</table>

We also added data reading/processing code for RACE ([convince/ParlAI/parlai/tasks/race/](https://github.com/ethanjperez/convince/tree/master/ParlAI/parlai/tasks/race)) and DREAM ([convince/ParlAI/parlai/tasks/dream/](https://github.com/ethanjperez/convince/tree/master/ParlAI/parlai/tasks/dream))

## Installation

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment (Python 3):

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```bash
    conda create -n convince_human python=3.6
    ```

3.  Activate the Conda environment

    ```bash
    conda activate convince_human
    ```

#### Installing the library and dependencies

Clone this repo and move to `convince/ParlAI/` (where all commands should be run from):
```bash
git clone https://github.com/ethanjperez/convince.git
cd convince/ParlAI
```

Install dependencies using `pip`:
```bash
pip install -r requirements.txt
```
You may also need to install [PyTorch 1.1](https://pytorch.org/) if you have dependency issues later on.

Link the cloned directory to your site-packages:
```bash
python setup.py develop
```

Any necessary data will be downloaded to `~/ParlAI/data`.

Now that you've installed ParlAI, follow these [instructions](https://github.com/ethanjperez/convince/blob/master/ParlAI/README_ParlAI.md#mturk) to setup and walk through ParlAI's MTurk functionality.

## Running Human Evaluation

To run human evaluation:
```bash
python parlai/mturk/tasks/context_evaluator/run.py \
  --dataset race \                      # Use evidence found on this dataset ('race' or 'dream')
  --prompt-type "quote and question" \  # Evidence evaluation setup: Evaluate single-sentence evidence
  --live                                # Without this flag, you'll run a debugging HIT in MTurk Sandbox without fees
```

We support the following evidence evaluation setups (via arguments to `--prompt-type`):
<table>
<tr>
    <td> <b> --prompt-type </b> </td>
    <td> <b> Evaluation Setup </b> </td>
</tr>
<tr>
    <td> 'question' </td>
    <td> Question-only baseline (no evidence shown) </td>
</tr>
<tr>
    <td> 'passage and question' </td>
    <td> Full passage baseline </td>
</tr>
<tr>
    <td> 'quote and question' </td>
    <td> Show one evidence sentence for one answer </td>
</tr>
<tr>
    <td> 'quotes and question' </td>
    <td> Show one evidence sentence for each answer (concatenated as a summary) </td>
</tr>
</table>

## Handling possible issues

Sometimes, you'll need to delete a set of HITs if launched evaluations are not cancelled properly (workers will email you that your HIT isn't working, though it was already cancelled). To do so, run:
```bash
python parlai/mturk/core/scripts/delete_hits.py
```

You can bonus a worker if they were not paid for a HIT (requires that a worker has completed a previous HIT of yours):
```bash
python parlai/mturk/core/scripts/bonus_workers.py --hit-id
```
You'll need to provide the HIT ID that you're bonusing.
Omit `--hit-id` if you have the Assignment ID instead of the HIT ID.
Try both (with or without `--hit-id`) if you have some ID related to the HIT but don't know if it's an Assignment ID or HIT ID.

We reject HITs very sparingly, as rejected HITs have major consequences for workers.
When we do reject HITs, it's usually because the worker was answering too quickly.
If you do give a rejection unfairly, the worker will likely email you, and you can ask for their HIT ID or Assignment ID (or perhaps find it in their email).
To reverse a rejected HIT that was given out unfairly, run the following code in Python:
```python
from parlai.mturk.core.mturk_manager import MTurkManager
manager = MTurkManager.make_taskless_instance()
# Run one of the below, depending on what ID you have. Try both if you don't know.
manager.approve_work('[INSERT ASSIGNMENT ID]', override_rejection=True)
manager.approve_assignments_for_hit('[INSERT HIT ID]', override_rejection=True)
```

## Evaluating your own evidence agents

Use the following steps to evaluate the evidence of your own trained agents:
- Run inference with your own agent 4 times total, with `--debate-mode` as (Ⅰ, Ⅱ, Ⅲ, or Ⅳ - once each).
- For each run, the code will save a log file of the form `debate_log.*json` in the save directory (whatever you specified after `--serialization-dir` during inference).
- Rename each file to `$DM.json`, where `$DM` specifies the `--debate-mode` you ran inference with to produce that file.
- Place the files in a new directory together within some directory `$DIR`
- Change the 'evaluation_data_dir' field value to `$DIR` in [task_configs.py](https://github.com/ethanjperez/convince/tree/master/ParlAI/parlai/mturk/tasks/context_evaluator/task_configs.py)
- Run human evaluation as described above
