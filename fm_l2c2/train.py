"""Run from the repository root: python -m fm_l2c2.train --help."""
from pathlib import Path

from fm_baseline import train as baseline
from .regularizer import L2C2Config, policy_regularization


class TrainingExtension:
    columns = ["fm_loss", "l2c2_loss", "weighted_l2c2_loss",
               "action_distance", "relative_distance"]

    def __init__(self, args):
        if args.dim is None:
            args.dim = 64 if args.task == "inserting" else 32
        if args.dim_mults is None:
            args.dim_mults = "1,2,4,8" if args.task == "inserting" else "1,4,8"
        self.config = L2C2Config(args.l2c2_weight, args.l2c2_sigma,
                                args.l2c2_epsilon, args.l2c2_batch_size,
                                args.l2c2_ode_steps, args.l2c2_action_steps)
        self.config.validate(args.horizon)
        if args.sn and self.config.weight:
            raise ValueError("SN is not part of the common baseline; extra forwards would update its buffers")
        if args.steps < 1 or args.batch_size < 1 or args.log_every < 1 or args.val_batches < 1:
            raise ValueError("steps, batch-size, log-every, val-batches must be positive")
        self.seed = args.seed

    def describe(self):
        return {"name": "l2c2_fm_shared_noise", "version": 1,
                **vars(self.config), "state_metric": "transition_relative_linf_plus_epsilon",
                "policy_metric": "half_squared_normalized_action_distance_mean_over_chunk",
                "source": "https://arxiv.org/abs/2202.07152", "paper_exact_reproduction": False}

    def __call__(self, model, windows, act_dim, step, fm_loss):
        loss, info = policy_regularization(model, windows, act_dim, self.config,
                                           seed=self.seed + 1000003 + step)
        weighted = self.config.weight * loss
        return weighted, [float(fm_loss.detach()), float(loss.detach()),
                          float(weighted.detach()), info["action_distance"],
                          info["relative_distance"]]


def configure(parser):
    parser.description = "L2C2-inspired shared-noise FM policy regularization (training required)"
    parser.add_argument("--l2c2-weight", type=float, default=0.01)
    parser.add_argument("--l2c2-sigma", type=float, default=1.)
    parser.add_argument("--l2c2-epsilon", type=float, default=0.1)
    parser.add_argument("--l2c2-batch-size", type=int, default=4)
    parser.add_argument("--l2c2-ode-steps", type=int, default=10)
    parser.add_argument("--l2c2-action-steps", type=int, default=8)
    # Artifacts stay in codex; checkpoint format remains compatible with baseline evaluation.
    parser.set_defaults(tag="l2c2", dim=None, dim_mults=None,
                        steps=1000000, scheduler="cosine", ema_decay=.995,
                        log_every=5000, save_every=25000,
                        out_root=str(Path(baseline.REPO) / "codex" / "l2c2" / "runs"))


def main(argv=None):
    args = baseline.parse_args(argv, configure)
    baseline.main(args, extension_factory=TrainingExtension)


if __name__ == "__main__":
    main()
