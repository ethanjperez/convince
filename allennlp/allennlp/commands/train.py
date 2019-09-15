"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ allennlp train --help

   usage: allennlp train [-h] -s SERIALIZATION_DIR [-r] [-f] [-o OVERRIDES]
                         [--file-friendly-logging]
                         [--include-package INCLUDE_PACKAGE]
                         param_path

   Train the specified model on the specified dataset.

   positional arguments:
     param_path            path to parameter file describing the model to be
                           trained

   optional arguments:
     -h, --help            show this help message and exit
     -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                           directory in which to save the model and its logs
     -r, --recover         recover training from the state in serialization_dir
     -f, --force           overwrite the output directory if it exists
     -o OVERRIDES, --overrides OVERRIDES
                           a JSON structure used to override the experiment
                           configuration
     --file-friendly-logging
                           outputs tqdm status on separate lines and slows tqdm
                           refresh rate
     --include-package INCLUDE_PACKAGE
                            additional packages to include

   debate arguments:
     -d, --debate_mode     List of debate turns (e.g. aa, ar, rr, Ar) => capital implies search agent
     -j, --judge_filename  Path to judge config or pre-trained judge model. If config, judge trained during debate
     -u, --update_judge    Boolean whether or not to update Judge model during debate training
     -m, --reward_method   Choice of Debate Reward function - [prob (judge probability on an answer option),
                               sl (Supervised Learning to predict search-chosen sentence),
                               sl-sents (SL to predict Judge answer prob if each sentence had been chosen,
                               sl-sents-influence (Like above, but predicting the change in Judge prob (not raw prob),
                               sl-random (Random Baseline: Use SL setup but with uniform random sentence selection]
     -v, --detach_val...   Boolean whether or not to detach value function from policy network to isolate gradients
     -b, --breakpoint...   Debugging option to set sensitivity of breakpoints
     -p, --search_outputs_path... Path to file with search predictions for supervised learning
     -a, --accumulation... Number of steps to accumulate gradient for before taking an optimizer step
     -e, --eval_mode       Boolean whether or not to run in eval-only mode, on test data
     -g, --multi-gpu       Boolean whether or not to load in model-parallel multi-gpu mode (allocation in config file)
     -c, --choice_mode     String type of action debating agents take
     -q, --qa_loss_weight  Float weight of auxiliary QA supervised loss to give RL agents
     -i, --influence       Boolean whether or not to use delta in judge opinion (vs. raw reward)
     Available flags: klwyz (v?)
"""
from typing import List
import argparse
import logging
import os
import warnings

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.common import Params
from allennlp.common.util import prepare_environment, prepare_global_logging, dump_metrics
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer, TrainerPieces
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.util import create_serialization_dir, evaluate

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Train(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(name, description=description, help='Train a model')

        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model to be trained')

        subparser.add_argument('-s', '--serialization-dir',
                               required=True,
                               type=str,
                               help='directory in which to save the model and its logs')

        subparser.add_argument('-r', '--recover',
                               action='store_true',
                               default=False,
                               help='recover training from the state in serialization_dir')

        subparser.add_argument('-f', '--force',
                               action='store_true',
                               required=False,
                               help='overwrite the output directory if it exists')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        # Debate options below
        subparser.add_argument('-d', '--debate-mode',
                               default=['f'],
                               nargs='+',
                               help='how to select sentences shown to judge')

        subparser.add_argument('-j', '--judge-filename',
                               type=str,
                               default=None,
                               help='path to parameter file describing the judge to be trained or'
                                    'path to an archived, trained judge.'
                                    'Do not use this option if training judge only.'
                                    'If updating pre-trained judge, pass in archived *.tar.gz file.'
                                    'If training judge from scratch, pass in *.jsonnet config file.')

        subparser.add_argument('-u', '--update-judge',
                               action='store_true',
                               default=False,
                               help='update judge while training debate agents')

        subparser.add_argument('-m', '--reward-method',
                               type=str,
                               choices=['prob',  # Judge probability on answer
                                        'sl',  # Supervised Learning (Oracle Prob)
                                        'sl-sents',  # SL on change in probabilities
                                        'sl-random'],  # SL but with random policy (baseline)
                               default='prob',
                               help='how to reward debate agents')

        subparser.add_argument('-v', '--detach-value-head',
                               action='store_true',
                               default=False,
                               help='Detach value head prediction network from main policy network,'
                                    'to prevent gradients to value function from overpowering gradients to policy')

        subparser.add_argument('-b', '--breakpoint-level',
                               type=int,
                               default=0,
                               help='Debugging option: Increase to run with more breakpoints. 0 for no breakpoints.')

        subparser.add_argument('-p', '--search-outputs-path',
                               type=str,
                               default=None,
                               help='Name file containing search predictions to do Supervised Learning on.')

        subparser.add_argument('-a', '--accumulation-steps',
                               type=int,
                               default=1,
                               help='Number of steps to accumulate gradient for before taking an optimizer step.')

        subparser.add_argument('-e', '--eval-mode',
                               action='store_true',
                               default=False,
                               help='run in evaluation-only mode on test_data_path (validation if no test given)')

        subparser.add_argument('-g', '--multi-gpu',
                               action='store_true',
                               default=False,
                               help='Run in model-parallel multiple GPU mode (gpu allocation in config file)')

        subparser.add_argument('-c', '--choice-mode',
                               type=str,
                               choices=['delete', 'reveal', 'concat'],
                               default='concat',
                               help='type of action debating agents take')

        subparser.add_argument('-q', '--qa-loss-weight',
                               type=float,
                               default=0.,
                               help='Weight of auxiliary QA supervised loss to give agents.')

        subparser.add_argument('-i', '--influence-reward',
                               action='store_true',
                               default=False,
                               help='Whether or not to use delta in judge opinion (vs. raw reward).')

        subparser.add_argument('-t', '--theory-of-mind',
                               action='store_true',
                               default=False,
                               help='Whether or not debaters use judge activations.')

        subparser.add_argument('-n', '--num-pred-rounds',
                               type=int,
                               default=-1,
                               help='Number of rounds debaters make predictions while training (vs. using search).'
                                    'If <1, debaters make prediction every round.')

        subparser.add_argument('-x', '--x-order-prob',
                               type=float,
                               default=0.,
                               help='Probability of reversing the debate agent ordering.')

        subparser.add_argument('-ra', '--require-action',
                               action='store_true',
                               default=False,
                               help='Whether or not debaters are required to choose a new sentence each turn.')

        subparser.add_argument('-ss', '--single-shot',
                               action='store_true',
                               default=False,
                               help='Debaters predict all turns\' sentence choices in a single shot.'
                                    'Only relevant for learned agents. Use only with -e (evaluation mode)')

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(args.param_path,
                          args.serialization_dir,
                          args.overrides,
                          args.file_friendly_logging,
                          args.recover,
                          args.force,
                          args.debate_mode,
                          args.judge_filename,
                          args.update_judge,
                          args.eval_mode,
                          args.reward_method,
                          args.detach_value_head,
                          args.breakpoint_level,
                          args.search_outputs_path,
                          args.accumulation_steps,
                          args.multi_gpu,
                          args.choice_mode,
                          args.qa_loss_weight,
                          args.influence_reward,
                          args.theory_of_mind,
                          args.num_pred_rounds,
                          args.x_order_prob,
                          args.require_action,
                          args.single_shot)

def train_model_from_file(parameter_filename: str,
                          serialization_dir: str,
                          overrides: str = "",
                          file_friendly_logging: bool = False,
                          recover: bool = False,
                          force: bool = False,
                          debate_mode: List[str] = ('f'),
                          judge_filename: str = None,
                          update_judge: bool = False,
                          eval_mode: bool = False,
                          reward_method: str = None,
                          detach_value_head: bool = False,
                          breakpoint_level: int = 0,
                          search_outputs_path: str = None,
                          accumulation_steps: int = 1,
                          multi_gpu: bool = False,
                          choice_mode: str = None,
                          qa_loss_weight: float = 0.,
                          influence_reward: bool = False,
                          theory_of_mind: bool = False,
                          num_pred_rounds: int = -1,
                          x_order_prob: float = 0.,
                          require_action: bool = False,
                          single_shot: bool = False) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    parameter_filename : ``str``
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`train_model`.
    debate_mode : ``List[str]``
        List of debate turns (e.g. aa, ar, rr, Ar) => capitalization implies search agent
    overrides : ``str``
        A JSON string that we will use to override values in the input parameter file.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`train_model`.
    recover : ``bool`, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    force : ``bool``, optional (default=False)
        If ``True``, we will overwrite the serialization directory if it already exists.
    judge_filename : ``str``, optional (default=None)
        Path to judge config or pre-trained judge model. If config, judge trained during debate. Necessary parameter
        if running in debate mode.
    update_judge : ``bool``, optional (default=False)
        Boolean whether or not to update Judge model during debate training.
    eval_mode : ``bool``, optional (default=False)
        Boolean whether or not to run in eval-only mode, on test data. Does not update/train any of the models.
    reward_method : ``str``, optional (default=False)
        Choice of reward function (RL) or loss function (Supervised Learning) for training debate agents
    detach_value_head : ``bool``, optional (default=False)
        Boolean whether or not to detatch value function gradient updates from the policy network. This prevents
        value function gradients from affecting policy network parameters.
    breakpoint_level : ``int`` optional (default=0)
        Debugging option to set breakpoint sensitivity (0 - no breakpoints).
    id_to_search_filename : ``str`` optional (default=None)
        Path to file with search predictions for each agent - necessary for supervised training
    accumulation_steps : ``int`` (default=1)
        Number of gradient steps to accumulate over before performing an update. Poor-man's batching for instances where
        number of examples per batch is small (limited GPU memory)
    multi_gpu : ``bool`` (default=False)
        Boolean whether or not to run models/training in model parallel mode. Requires specifying GPU allocations for
        trainer, judge, and debaters in the training config file.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(params, serialization_dir, file_friendly_logging, recover, force, debate_mode, judge_filename,
                       update_judge, eval_mode, reward_method, detach_value_head, breakpoint_level,
                       search_outputs_path, accumulation_steps, multi_gpu, choice_mode, qa_loss_weight,
                       influence_reward, theory_of_mind, num_pred_rounds, x_order_prob, require_action, single_shot)


def train_model(params: Params,
                serialization_dir: str,
                file_friendly_logging: bool = False,
                recover: bool = False,
                force: bool = False,
                debate_mode: List[str] = ('f'),
                judge_filename: str = None,
                update_judge: bool = False,
                eval_mode: bool = False,
                reward_method: str = None,
                detach_value_head: bool = False,
                breakpoint_level: int = 0,
                search_outputs_path: str = None,
                accumulation_steps: int = 1,
                multi_gpu: bool = False,
                choice_mode: str = None,
                qa_loss_weight: float = 0.,
                influence_reward: bool = False,
                theory_of_mind: bool = False,
                num_pred_rounds: int = -1,
                x_order_prob: float = 0.,
                require_action: bool = False,
                single_shot: bool = False) -> Model:
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    debate_mode : ``List[str]``
        List of debate turns (e.g. aa, ar, rr, Ar) => capitalization implies search agent
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    force : ``bool``, optional (default=False)
        If ``True``, we will overwrite the serialization directory if it already exists.
    judge_filename : ``str``, optional (default=None)
        Path to judge config or pre-trained judge model. If config, judge trained during debate. Necessary parameter
        if running in debate mode.
    update_judge : ``bool``, optional (default=False)
        Boolean whether or not to update Judge model during debate training.
    eval_mode : ``bool``, optional (default=False)
        Boolean whether or not to run in eval-only mode, on test data. Does not update/train any of the models.
    reward_method : ``str``, optional (default=False)
        Choice of reward function (RL) or loss function (Supervised Learning) for training debate agents
    detach_value_head : ``bool``, optional (default=False)
        Boolean whether or not to detatch value function gradient updates from the policy network. This prevents
        value function gradients from affecting policy network parameters.
    breakpoint_level : ``int`` optional (default=0)
        Debugging option to set breakpoint sensitivity (0 - no breakpoints).
    id_to_search_filename : ``str`` optional (default=None)
        Path to file with search predictions for each agent - necessary for supervised training
    accumulation_steps : ``int`` (default=1)
        Number of gradient steps to accumulate over before performing an update. Poor-man's batching for instances where
        number of examples per batch is small (limited GPU memory)
    multi_gpu : ``bool`` (default=False)
        Boolean whether or not to run models/training in model parallel mode. Requires specifying GPU allocations for
        trainer, judge, and debaters in the training config file (see training_config/bidaf.race.size=0.5.gpu=2.jsonnet
        for example usage).

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    assert (not single_shot) or eval_mode, 'Using single shot prediction outside eval_mode not yet supported.'
    assert (not single_shot) or (num_pred_rounds == -1), \
        'Using single shot prediction for a specific number of rounds is not yet supported.'
    # Get number of debate turns, and assert that not performing judge-only training
    num_no_qa_turns = sum([(('l' in debate_turn) or ('w' in debate_turn)) for debate_turn in debate_mode])
    if (qa_loss_weight > 0) and (num_no_qa_turns == 0):
        warnings.warn('Unused argument qa_loss_weight in debate mode ' + str(debate_mode) +
                      '. If this was unintentional, please remove the -q flag.', UserWarning)
    not_using_trained_debater = len(set('ablwⅰⅱⅲⅳ').intersection(''.join(debate_mode))) == 0
    if (judge_filename is not None) and not_using_trained_debater:
        warnings.warn('Unnecessary to have debaters in debate mode ' + str(debate_mode) +
                      '. If this was unintentional, please remove the -j flag.', UserWarning)

    prepare_environment(params)
    create_serialization_dir(params, serialization_dir, recover, force)
    prepare_global_logging(serialization_dir, file_friendly_logging)

    # Check that all Desired CUDA Devices exist => trainer => cuda_devices should contain list of required devices
    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    check_for_gpu(cuda_device)

    # Build Allocation Dictionary (to be passed to all future functions)
    if multi_gpu:
        gpu_allocations, allocation_dict = params.params.pop('gpu_allocations', {}), {}
        assert len(gpu_allocations) == 3, 'Must set gpu_allocations in config if multi-gpu'
        for k in ['debate', 'judge', 'trainer']:
            assert gpu_allocations[k] in cuda_device, "Desired GPU not available... current: %s" % str(cuda_device)
            allocation_dict[k] = gpu_allocations[k]
    else:
        allocation_dict = {}

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)

    trainer_type = params.get("trainer", {}).get("type", "default")

    if trainer_type == "default":
        # Special logic to instantiate backward-compatible trainer.
        params['dataset_reader']['debate_mode'] = debate_mode  # If debate_mode requires sample duplicates
        pieces = TrainerPieces.from_params(params,
                                           serialization_dir,
                                           cuda_device,
                                           recover,
                                           judge_filename=judge_filename,
                                           update_judge=update_judge,
                                           eval_mode=eval_mode,
                                           reward_method=reward_method,
                                           detach_value_head=detach_value_head,
                                           allocation_dict=allocation_dict,
                                           qa_loss_weight=qa_loss_weight,
                                           influence_reward=influence_reward,
                                           theory_of_mind=theory_of_mind)  # pylint: disable=no-member
        trainer = Trainer.from_params(
                model=pieces.model,
                serialization_dir=serialization_dir,
                debate_mode=debate_mode,
                iterator=pieces.iterator,
                train_data=pieces.train_dataset,
                validation_data=pieces.validation_dataset,
                params=pieces.params,
                validation_iterator=pieces.validation_iterator,
                eval_mode=eval_mode,
                breakpoint_level=breakpoint_level,
                search_outputs_path=search_outputs_path,
                accumulation_steps=accumulation_steps,
                allocation_dict=allocation_dict,
                choice_mode=choice_mode,
                num_pred_rounds=num_pred_rounds,
                x_order_prob=x_order_prob,
                require_action=require_action,
                single_shot=single_shot)
        evaluation_iterator = pieces.validation_iterator or pieces.iterator
        evaluation_dataset = pieces.test_dataset
    else:
        assert (len(debate_mode) == 1) and (debate_mode[0] == 'f'), 'TrainerBase untested for debate training.'
        trainer = TrainerBase.from_params(params, serialization_dir, recover)
        evaluation_iterator = evaluation_dataset = None

    params.assert_empty('base train command')

    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)) and not eval_mode:
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    # Evaluate
    if evaluation_dataset and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(trainer.model, evaluation_dataset, evaluation_iterator,
                                cuda_device=trainer._cuda_devices[0], # pylint: disable=protected-access,
                                batch_weight_key="")

        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif evaluation_dataset:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    # Now tar up results
    if not eval_mode:
        archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)
    else:
        dump_metrics(os.path.join(serialization_dir, "metrics.eval.d=" + '-'.join(debate_mode) + ".json"), metrics,
                     log=True)

    # We count on the trainer to have the model with best weights
    return trainer.model
