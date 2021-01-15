# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

import main as detection
import submitit


def parse_args():
    detection_parser = detection.get_args_parser()
    parser = argparse.ArgumentParser(
        "Submitit for detection", parents=[detection_parser]
    )
    parser.add_argument(
        "--ngpus", default=4, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes to request"
    )
    parser.add_argument(
        "--hours", default=12, type=int, help="Duration of the job in hours"
    )
    parser.add_argument(
        "--job_dir", default="", type=str, help="Job dir. Leave empty for automatic."
    )
    parser.add_argument("--experiment_name", type=str, default="test_experiment_delete")
    return parser.parse_args()


def get_shared_folder(experiment_name) -> Path:
    if Path("/work/").is_dir():
        p = Path(f"/work/korbar/DETR_experiments/{experiment_name}")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file(experiment_name):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(experiment_name)), exist_ok=True)
    init_file = get_shared_folder(experiment_name) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as detection

        self._setup_gpu_args()
        detection.main(self.args)

    def checkpoint(self):
        import os
        import submitit
        from pathlib import Path

        self.args.dist_url = get_init_file(self.args.experiment_name).as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(
            str(self.args.output_dir).replace("%j", str(job_env.job_id))
        )
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder(args.experiment_name) / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    # cluster setup is defined by environment variables
    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.hours * 60

    executor.update_parameters(
        mem_gb=20 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=8,
        nodes=nodes,
        slurm_partition="gpu",
        timeout_min=timeout_min,  # max is 60 * 72
    )

    executor.update_parameters(name=f"detr_{args.experiment_name}")

    args.dist_url = get_init_file(args.experiment_name).as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
