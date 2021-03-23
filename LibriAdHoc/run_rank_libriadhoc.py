import argparse
import numpy as np
import os
import random
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from local.rank_trainer import MicRank

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
import torchaudio
import torch
from micrank.metrics import compute_nwer, compute_wer

parser = argparse.ArgumentParser("Training a MicRank system for LibriAdhoc dataset")
parser.add_argument("--conf_file", default="./confs/libriadhoc.yaml",
                    help="The configuration file with all the experiment parameters.")
parser.add_argument("--log_dir", default="./exp/tcn",
                    help="Directory where to save tensorboard logs, saved models, etc.")
parser.add_argument("--resume_from_checkpoint", default=None,
                    help="Allow the training to be resumed, take as input a previously saved model (.ckpt).")
parser.add_argument("--test_from_checkpoint", default=None,
                    help="Test the model specified")
parser.add_argument("--gpus", default="0", help="The number of GPUs to train on, or the gpu to use, default='0', "
                                                "so uses one GPU indexed by 0.")
parser.add_argument("--fast_dev_run", action="store_true", default=False,
                    help="Use this option to make a 'fake' run which is useful for development and debugging. "
                         "It uses very few batches and epochs so it won't give any meaningful result.")


def get_datasets(hparams):

    train_set = DynamicItemDataset.from_json(hparams["data"]["train_json"])
    dev_set = DynamicItemDataset.from_json(hparams["data"]["dev_json"])
    test_set = DynamicItemDataset.from_json(hparams["data"]["test_json"])
    if hparams["training"]["delta_wer"]:
        # we dump examples with wer difference less than X

        new_ids = []
        new_data = {}
        for k in train_set.data_ids:
            wers = []
            if train_set.data[k]["n_words"] == 0:
                continue
            for ch in train_set.data[k]["channels"]:
                wer = compute_wer(ch["csid"][1], ch["csid"][-1], ch["csid"][-2],  train_set.data[k]["n_words"])
                wers.append(wer)
            delta = hparams["training"]["delta_wer"]
            if not np.all([(wers[0] - delta) < x < (wers[0] + delta) for x in wers]): # discard whole utterance if
                if train_set.data[k]["length"] >= hparams["training"]["discard_shorter"]:
                    new_ids.append(k)
                    new_data[k] = train_set.data[k]
        print("N examples in training set {}, discarded: {}".format(len(new_ids), len(train_set.data_ids) - len(new_ids)))
        train_set.data_ids = new_ids
        train_set.data = new_data

    if hparams["training"]["sorted"] == "ascending":
        # we filter based on words
        train_set = train_set.filtered_sorted(sort_key="length")
    elif hparams["training"]["sorted"] == "descending":
        train_set = train_set.filtered_sorted(sort_key="length", reverse=True)
    elif hparams["training"]["sorted"] == "random":
        pass
    else:
        raise NotImplementedError

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("audio_file")
    @sb.utils.data_pipeline.provides("audio")
    def audio_pipeline(wav):
        sig, fs = torchaudio.load(wav)
        # we do shift augmentation
        for ch in range(sig.shape[0]):
            rolling = np.random.randint(0, sig.shape[-1])
            sig[ch] = torch.roll(sig[ch], rolling, dims=0)

        return sig

    if hparams["training"]["rankscore"] == "nwer":
        @sb.utils.data_pipeline.takes("channels", "n_words")
        @sb.utils.data_pipeline.provides("scores")
        def targets_pipeline(channels, n_words):
            targets = []
            for ch in channels:
                nwer = compute_nwer(ch["csid"][1], ch["csid"][-1], ch["csid"][-2],  n_words)
                targets.append(nwer)

            return torch.Tensor(targets).float()
    else:
        raise NotImplementedError

    @sb.utils.data_pipeline.takes("channels")
    @sb.utils.data_pipeline.provides("csid", "file_ids")
    def csid_pipeline(channels):
        file_ids = []
        csid = []
        for ch in channels:
            csid.append(ch["csid"])
            file_ids.append("dummy")
        yield csid
        yield file_ids

    sb.dataio.dataset.add_dynamic_item([train_set, dev_set, test_set], audio_pipeline)
    sb.dataio.dataset.add_dynamic_item([train_set, dev_set, test_set], targets_pipeline)
    sb.dataio.dataset.add_dynamic_item([train_set, dev_set, test_set], csid_pipeline)

    sb.dataio.dataset.set_output_keys([train_set, dev_set, test_set], ["id", "audio", "scores", "csid", "n_words", "file_ids"])

    return train_set, dev_set, test_set


def single_run(
    config,
    log_dir,
    gpus,
    checkpoint_resume=None,
    test_state_dict=None,
    fast_dev_run=False,
):
    """
    Running a ranking experiment
    Args:
        config (dict): the dictionary of configuration params
        log_dir (str): path to log directory
        gpus (int): number of gpus to use
        checkpoint_resume (str, optional): path to checkpoint to resume from. Defaults to "".
        test_state_dict (dict, optional): if not None, no training is involved. This dictionary is the state_dict
            to be loaded to test the model.
        fast_dev_run (bool, optional): whether to use a run with only one batch at train and validation, useful
            for development purposes.
    """
    config.update({"log_dir": log_dir})

    train_set, dev_set, test_set = get_datasets(config)

    ##### Model Definition  ############
    from micrank.rankers.TCN import TCN
    from micrank.rankers.transformer import Transformer_SC
    from micrank.rankers.CRNN import CRNN
    from micrank.rankers.dsp_selection import EnvelopeVariance, CepstralDistance

    ranker = TCN(**config["tcn"])#CRNN(**config["crnn"])

    if test_state_dict is None:
        opt = torch.optim.SGD(ranker.parameters(), **config["opt"])

        logger = TensorBoardLogger(
            os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1],
        )
        print(f"experiment dir: {logger.log_dir}")

        callbacks = [
            EarlyStopping(
                monitor="val/wer",
                patience=config["training"]["early_stop_patience"],
                verbose=True,
                mode="min"
            ),
            ModelCheckpoint(logger.log_dir, monitor="val/wer", save_top_k=1, mode="min",
                            save_last=True),
        ]

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                               factor=config["scheduler"]["factor"],
                                                               patience=config["scheduler"]["patience"],
                                                               verbose=True),
            "interval": "epoch",
            "monitor": "val/wer"
        }

    else:
        train_set = None
        dev_set = None
        test_set = None
        opt = None
        scheduler = None
        logger = True
        callbacks = None

    ranking_sys = MicRank(
        config,
        train_data=train_set,
        valid_data=dev_set,
        test_data=test_set,
        rank_model=ranker,
        optimizer=opt,
        scheduler=scheduler,
        fast_dev_run=fast_dev_run,
    )

    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    if fast_dev_run:
        flush_logs_every_n_steps = 1
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
        n_epochs = 3
    else:
        flush_logs_every_n_steps = 100
        log_every_n_steps = 40
        limit_train_batches = 1.
        limit_val_batches = 1.
        limit_test_batches = 1.
        n_epochs = config["training"]["n_epochs"]

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=callbacks,
        gpus=gpus,
        distributed_backend=config["training"].get("backend"),
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        resume_from_checkpoint=checkpoint_resume,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        log_every_n_steps=log_every_n_steps,
        flush_logs_every_n_steps=flush_logs_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        num_sanity_val_steps=0
    )

    if test_state_dict is None:
        trainer.fit(ranking_sys)
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]

    ranking_sys.load_state_dict(test_state_dict)
    trainer.test(ranking_sys)


if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    test_from_checkpoint = args.test_from_checkpoint
    test_model_state_dict = None
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint)
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        configs = configs_ckpt
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]

    seed = configs["training"]["seed"]
    if seed:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)

    test_only = test_from_checkpoint is not None
    single_run(
        configs,
        args.log_dir,
        args.gpus,
        args.resume_from_checkpoint,
        test_model_state_dict,
        args.fast_dev_run,
    )
