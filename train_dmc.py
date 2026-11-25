''' An example of training a Deep Monte-Carlo (DMC) Agent on the environments in RLCard
'''
import os
import argparse

import torch

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer

def train(args):

    # Make the environment
    env_config = (
        {'legacy_obs': args.legacy_obs, 'win_reward_weight': args.win_reward_weight}
        if args.env == 'swy-blm'
        else {}
    )
    env = rlcard.make(args.env, config=env_config)

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
        mlp_layers=[256,256,256,128],
        augment_symmetry=not args.no_augment,
        aug_copies=args.aug_copies,
        loss_type=args.loss_type,
        loss_beta=args.loss_beta,
        snapshot_prob=args.snapshot_prob,
        rule_opponent_prob=args.rule_opponent_prob,
        snapshot_pool_size=args.snapshot_pool_size,
        snapshot_interval=args.snapshot_interval,
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
    parser.add_argument(
        '--no_augment',
        action='store_true',
        help='Disable suit/color symmetry augmentation during training',
    )
    parser.add_argument(
        '--aug_copies',
        default=2,
        type=int,
        help='Number of symmetry-augmented copies per batch (swy-blm)',
    )
    parser.add_argument(
        '--legacy_obs',
        action='store_true',
        help='Use legacy 230-dim state representation for swy-blm',
    )
    parser.add_argument(
        '--win_reward_weight',
        default=5.0,
        type=float,
        help='Win/loss bonus weight added to score_diff in swy-blm payoffs (default: 5.0)',
    )
    parser.add_argument(
        '--loss_type',
        default='smooth_l1',
        choices=['mse', 'smooth_l1'],
        help='Regression loss type (default: smooth_l1)',
    )
    parser.add_argument(
        '--loss_beta',
        default=2.0,
        type=float,
        help='Beta parameter for smooth_l1 loss (default: 2.0)',
    )
    parser.add_argument(
        '--snapshot_prob',
        default=0.10,
        type=float,
        help='Fraction of actor games played against historical snapshots (default: 0.10)',
    )
    parser.add_argument(
        '--rule_opponent_prob',
        default=0.10,
        type=float,
        help='Fraction of actor games played against RuleBasedAgent on swy-blm (default: 0.10)',
    )
    parser.add_argument(
        '--snapshot_pool_size',
        default=8,
        type=int,
        help='Maximum number of historical snapshots kept in pool (default: 8)',
    )
    parser.add_argument(
        '--snapshot_interval',
        default=1000,
        type=int,
        help='Learner steps between saving a snapshot into the pool (default: 1000)',
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

