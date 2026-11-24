''' An example of training a Deep Monte-Carlo (DMC) Agent on the environments in RLCard
'''
import os
import argparse

import torch

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer

def train(args):

    # Make the environment
    env = rlcard.make(args.env)

    # Initialize the DMC trainer
    trainer = DMCTrainer(
        env,
        cuda=args.cuda,
        actor_device=args.actor_device,
        load_model=args.load_model,
        xpid=args.xpid,
        savedir=args.savedir,
        save_interval=args.save_interval,
        total_iterations=args.total_iterations,
        num_actor_devices=args.num_actor_devices,
        num_actors=args.num_actors,
        training_device=args.training_device,
        learning_rate=args.learning_rate,
        exp_epsilon=args.exp_epsilon,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        num_buffers=args.num_buffers,
        num_threads=args.num_threads,
        sync_interval=args.sync_interval,
        mlp_layers=[256,256,256,128]
    )

    # Train DMC Agents
    trainer.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DMC example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'swy-blm'
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--load_model',
        action='store_true',
        help='Load an existing model',
    )
    parser.add_argument(
        '--xpid',
        default='leduc_holdem',
        help='Experiment id (default: leduc_holdem)',
    )
    parser.add_argument(
        '--savedir',
        default='results/dmc_result',
        help='Root dir where experiment data will be saved'
    )
    parser.add_argument(
        '--save_interval',
        default=30,
        type=int,
        help='Time interval (in minutes) at which to save the model',
    )
    parser.add_argument(
        '--total_iterations',
        default=2000000,
        type=int,
        help='Total iterations',
    )
    parser.add_argument(
        '--num_actor_devices',
        default=1,
        type=int,
        help='The number of devices used for simulation',
    )
    parser.add_argument(
        '--num_actors',
        default=5,
        type=int,
        help='The number of actors for each simulation device',
    )
    parser.add_argument(
        '--training_device',
        default="0",
        type=str,
        help='The index of the GPU used for training models',
    )
    parser.add_argument(
        '--actor_device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device used for simulation actors: "cpu" (recommended for small MLPs) or "cuda"',
    )
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Learner batch size',
    )
    parser.add_argument(
        '--unroll_length',
        default=10,
        type=int,
        help='Unroll length of trajectory segments',
    )
    parser.add_argument(
        '--num_buffers',
        default=200,
        type=int,
        help='Number of shared-memory buffers',
    )
    parser.add_argument(
        '--num_threads',
        default=1,
        type=int,
        help='Number of learner threads per position',
    )
    parser.add_argument(
        '--sync_interval',
        default=10,
        type=int,
        help='Learner steps between actor weight synchronizations',
    )
    parser.add_argument(
        '--learning_rate',
        default=0.0002,
        type=float,
        help='Learning rate',
    )
    parser.add_argument(
        '--exp_epsilon',
        default=0.01,
        type=float,
        help='Epsilon-greedy exploration rate during actor simulation',
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

