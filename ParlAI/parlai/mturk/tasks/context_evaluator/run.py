#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.context_evaluator.worlds import ContextEvaluationWorld
from parlai.mturk.tasks.context_evaluator.worlds_onboard import ContextEvaluationOnboardWorld
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.core import shared_utils
from parlai.mturk.tasks.context_evaluator.task_configs import task_configs
from pprint import pprint
import datetime
import os
import importlib
import json

MASTER_QUALIF = {
    'QualificationTypeId': '2F1QJWKUDD8XADTFD2Q0G6UTO95ALH',
    'Comparator': 'Exists',
    'RequiredToPreview': True
}


def main():
    """Handles setting up and running a ParlAI-MTurk task by instantiating
    an MTurk manager and configuring it for the context_evaluator task
    """
    # Get relevant arguments
    shared_utils.disable_logging()
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_context_evaluation_args()
    opt = argparser.parse_args(print_args=False)

    # Set the task name to be the folder name
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

    # Append the contents of task_config.py to the configuration
    opt.update(task_configs['general'])
    opt.update(task_configs[opt['dataset']])
    opt.update(task_configs['sandbox' if opt['is_sandbox'] else 'live'])
    opt.update(task_configs[opt['prompt_type']])
    pprint(opt)
    if opt['evaluation_data_dir'] is not None:
        assert opt['dataset'] in opt['evaluation_data_dir'], 'Dataset name must be in evaluation data dir name.'

    # Load data to evaluate
    evaluation_data = None
    if opt['evaluation_data_dir'] is not None:
        evaluation_data = {}
        for filename in os.listdir(opt['evaluation_data_dir']):
            with open(os.path.join(opt['evaluation_data_dir'], filename)) as json_file:
                evaluation_data[filename[:-5]] = json.load(json_file)

    # Track which evaluation splits have been / are being worked on
    global active_workers_per_incomplete_hit_by_split, active_workers_by_split, incomplete_hits_by_split
    # The values in these maps should always be non-negative
    active_workers_per_incomplete_hit_by_split, active_workers_by_split, incomplete_hits_by_split = {}, {}, {}
    for q_spl in range(opt['question_splits']):
        option_splits = opt['num_options'] if opt['prompt_type'] in {'quote and question'} else 1
        for o_spl in range(option_splits):
            active_workers_by_split[(q_spl, o_spl)] = 0
            incomplete_hits_by_split[(q_spl, o_spl)] = opt['num_conversations'] / (
                    opt['question_splits'] * option_splits)
            active_workers_per_incomplete_hit_by_split[(q_spl, o_spl)] = (
                    active_workers_by_split[(q_spl, o_spl)] / incomplete_hits_by_split[(q_spl, o_spl)])

    # Track stats about how many people have passed onboarding
    global num_passed_agents, num_total_agents
    num_passed_agents, num_total_agents = 0, 0

    # Initialize a dataset agent, which we will get quote from
    task_class = getattr(importlib.import_module('parlai.tasks.' + opt['dataset'] + '.agents'), 'IndexTeacher')
    task_opt = opt.copy()

    # Instantiate an MTurkManager with the given options and a maximum number
    # of agents per world of 1 (based on the length of mturk_agent_ids)
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=[opt['mturk_agent_id']],
        use_db=True,
    )
    mturk_manager.setup_server()  # Can pass in os.path.dirname(os.path.abspath(__file__))

    # Create an onboard_function, which will be run for workers who have
    # accepted your task and must be completed before they are put in the
    # queue for a task world.
    def run_onboard(worker):
        world = ContextEvaluationOnboardWorld(opt=opt, mturk_agent=worker)
        while not world.episode_done():
            world.parley()
        if world.passed_test is not None:
            global num_passed_agents
            num_passed_agents += world.passed_test
            global num_total_agents
            num_total_agents += 1
            print('TEST PASS RATE:', num_passed_agents, '/', num_total_agents)
        world.shutdown()
        return world.prep_save_data([worker])

    # If we want to use the above onboard function, we can replace the below
    # with set_onboard_function(onboard_function=run_onboard) (onboard_function=None to skip)
    mturk_manager.set_onboard_function(onboard_function=run_onboard)

    try:
        # Initialize run information
        mturk_manager.start_new_run()

        # Set up the sockets and threads to recieve workers
        mturk_manager.ready_to_accept_workers()

        # Create the hits as specified by command line arguments
        mturk_manager.create_hits()

        # Check workers eligiblity acts as a filter, and should return
        # the list of all workers currently eligible to work on the task
        def check_workers_eligibility(workers):
            return workers

        eligibility_function = {
            'func': check_workers_eligibility,
            'multiple': True,
        }

        # Assign worker roles is used to determine what the role each worker
        # in the given worker list will play. Setting `id` to None will return
        # the worker to the pool rather than putting them in a given task,
        # which is useful for having tasks with different possible worker
        # counts.
        def assign_worker_roles(workers):
            workers[0].id = task_opt['mturk_agent_id']

        # Define the task function, which will be run with workers that are
        # as the main task.
        global run_conversation

        def run_conversation(mturk_manager, opt, workers):
            # create a task agent to ask the questions
            q_spl, o_spl = min(active_workers_per_incomplete_hit_by_split,
                               key=active_workers_per_incomplete_hit_by_split.get)
            active_workers_by_split[(q_spl, o_spl)] += 1
            active_workers_per_incomplete_hit_by_split[(q_spl, o_spl)] = (
                    active_workers_by_split[(q_spl, o_spl)] / incomplete_hits_by_split[(q_spl, o_spl)])
            task_opt['question_split_no'] = q_spl
            task_opt['option_split_no'] = o_spl
            opt['question_split_no'] = q_spl
            opt['option_split_no'] = o_spl
            print('Worker starting...')
            print('active_workers_by_split:', active_workers_by_split)
            print('incomplete_hits_by_split:', incomplete_hits_by_split)
            print('active_workers_per_incomplete_hit_by_split:', active_workers_per_incomplete_hit_by_split)

            task = task_class(task_opt)
            # Create the task world
            world = ContextEvaluationWorld(
                opt=opt,
                task=task,
                mturk_agent=workers[0],
                evaluation_data=evaluation_data
            )
            # run the world to completion
            while not world.episode_done():
                world.parley()

            # shutdown and review the work
            world.shutdown()
            world.review_work()

            active_workers_by_split[(q_spl, o_spl)] = max(0, active_workers_by_split[(q_spl, o_spl)] - 1)
            if world.hit_done and (len(world.reject_reasons) == 0):
                incomplete_hits_by_split[(q_spl, o_spl)] = max(0, incomplete_hits_by_split[(q_spl, o_spl)] - 1)
            active_workers_per_incomplete_hit_by_split[(q_spl, o_spl)] = (
                    float('inf') if incomplete_hits_by_split[(q_spl, o_spl)] <= 0 else
                    active_workers_by_split[(q_spl, o_spl)] / incomplete_hits_by_split[(q_spl, o_spl)])
            print('Worker finishing...')
            print('active_workers_by_split:', active_workers_by_split)
            print('incomplete_hits_by_split:', incomplete_hits_by_split)
            print('active_workers_per_incomplete_hit_by_split:', active_workers_per_incomplete_hit_by_split)
            if max(list(incomplete_hits_by_split.values())) <= 0:
                print('********** COMPLETED HITS! **********', str(datetime.datetime.now()))
                mturk_manager.completed_conversations = mturk_manager.num_conversations  # Signal no more HITs needed

            # Return the contents for saving
            return world.prep_save_data(workers)

        # Begin the task, allowing mturk_manager to start running the task
        # world on any workers who connect
        mturk_manager.start_task(
            eligibility_function=eligibility_function,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )
    except BaseException:
        raise
    finally:
        # Any hits that aren't claimed or completed have to be shut down. Must
        # keep the world running until that point.
        mturk_manager.expire_all_unassigned_hits()
        # Shutdown the manager and free all related resources
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()
