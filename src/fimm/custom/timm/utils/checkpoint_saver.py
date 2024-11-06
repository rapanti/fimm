"""Checkpoint Saver

Track top-n training checkpoints and maintain recovery checkpoints on specified intervals.

Modified from timm.utils.checkpoint_saver.py.
"""

import glob
import operator
import os
import logging
import pickle
import torch

from timm.utils.model import unwrap_model, get_state_dict


_logger = logging.getLogger(__name__)


class CheckpointSaver:
    """
    Track top-n training checkpoints and maintain recovery checkpoints on specified intervals.

    Modified from timm.utils.checkpoint_saver.py.
    """

    def __init__(
        self,
        model,
        optimizer,
        args=None,
        model_ema=None,
        amp_scaler=None,
        checkpoint_prefix="checkpoint",
        recovery_prefix="recovery",
        checkpoint_dir="",
        recovery_dir="",
        decreasing=False,
        max_history=10,
        unwrap_fn=unwrap_model,
        resume=False,
    ):
        """
        Initialize CheckpointSaver.

        Args:
            model: Model to be saved
            optimizer: Optimizer to be saved
            args: Arguments to be saved
            model_ema: EMA model to be saved
            amp_scaler: AMP scaler to be saved
            checkpoint_prefix: Prefix for checkpoint filenames
            recovery_prefix: Prefix for recovery filenames
            checkpoint_dir: Directory for saving checkpoints
            recovery_dir: Directory for saving recoveries
            decreasing: Whether to use decreasing metric for sorting checkpoints
            max_history: Maximum number of checkpoints to keep
            unwrap_fn: Function to unwrap model before saving
            resume: Whether to restore from previous checkpoint saver state
        """
        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_ema = model_ema
        self.amp_scaler = amp_scaler

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ""
        self.last_recovery_file = ""

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = ".pth.tar"
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt
        self.max_history = max_history
        self.unwrap_fn = unwrap_fn
        assert self.max_history >= 1

        if resume:
            state_path = os.path.join(self.checkpoint_dir, "checkpoint_saver.pkl")
            try:
                with open(state_path, "rb") as f:
                    state_saver = pickle.load(f)
                self.__dict__.update(state_saver)
            except FileNotFoundError:
                _logger.warning(
                    "Resume requested but CheckpointSaver state-file not found. "
                    "This might lead to enexpected behavior. "
                    "Best model must not be the best model."
                )
            _logger.info("Restored CheckpointSaver state from '{}'".format(state_path))

    def save_checkpoint(self, epoch, metric=None):
        """
        Save checkpoint.

        Args:
            epoch: Epoch to be saved
            metric: Metric to be saved
        """
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, "tmp" + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, "last" + self.extension)
        self._save(tmp_save_path, epoch, metric)
        if os.path.exists(last_save_path):
            os.unlink(last_save_path)  # required for Windows support.
        os.rename(tmp_save_path, last_save_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (
            len(self.checkpoint_files) < self.max_history
            or metric is None
            or self.cmp(metric, worst_file[1])  # type: ignore
        ):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = "-".join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            os.link(last_save_path, save_path)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1], reverse=not self.decreasing
            )  # sort in descending order if a lower metric is not better

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += " {}\n".format(c)
            _logger.info(checkpoints_str)

            if metric is not None and (
                self.best_metric is None or self.cmp(metric, self.best_metric)
            ):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, "model_best" + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_save_path, best_save_path)
        # save checkpoint saver state
        self._save_self_state()

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, epoch, metric=None):
        """
        Save checkpoint to file.

        Args:
            save_path: Path to save checkpoint
            epoch: Epoch to be saved
            metric: Metric to be saved
        """
        save_state = {
            "epoch": epoch,
            "arch": type(self.model).__name__.lower(),
            "state_dict": get_state_dict(self.model, self.unwrap_fn),
            "optimizer": self.optimizer.state_dict(),
            "version": 2,  # version < 2 increments epoch before save
        }
        if self.args is not None:
            save_state["arch"] = self.args.model
            save_state["args"] = self.args
        if self.amp_scaler is not None:
            save_state[self.amp_scaler.state_dict_key] = self.amp_scaler.state_dict()
        if self.model_ema is not None:
            save_state["state_dict_ema"] = get_state_dict(self.model_ema, self.unwrap_fn)
        if metric is not None:
            save_state["metric"] = metric
        torch.save(save_state, save_path)

    def _save_self_state(self):
        """
        Save CheckpointSaver state to file.
        """
        # save checkpoint saver state
        state_path = os.path.join(self.checkpoint_dir, "checkpoint_saver.pkl")
        state_saver = {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "checkpoint_files": self.checkpoint_files,
            "curr_recovery_file": self.curr_recovery_file,
            "last_recovery_file": self.last_recovery_file,
        }
        with open(state_path, "wb") as f:
            pickle.dump(state_saver, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _cleanup_checkpoints(self, trim=0):
        """
        Cleanup checkpoints.

        Args:
            trim: Number of checkpoints to keep
        """
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                _logger.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                _logger.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, epoch, batch_idx=0):
        """
        Save recovery checkpoint.

        Args:
            epoch: Epoch to be saved
            batch_idx: Batch index to be saved
        """
        assert epoch >= 0
        filename = "-".join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, epoch)
        if os.path.exists(self.last_recovery_file):
            try:
                _logger.debug("Cleaning recovery: {}".format(self.last_recovery_file))
                os.remove(self.last_recovery_file)
            except Exception as e:
                _logger.error("Exception '{}' while removing {}".format(e, self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        """
        Find the latest recovery checkpoint.

        Returns:
            Path to the latest recovery checkpoint
        """
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + "*" + self.extension)
        files = sorted(files)
        return files[0] if len(files) else ""
