import pytorch_lightning as pl
import torch
import torchaudio
from micrank.utils.batch import PaddedBatch
import numpy as np
from micrank.metrics import compute_wer
from micrank.utils.scaler import TorchScaler
import os

class MicRank(pl.LightningModule):

    def __init__(self, hparams, train_data, valid_data, test_data, rank_model, optimizer, scheduler, fast_dev_run):
        super(MicRank, self).__init__()
        self.hparams = hparams
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.ranker = rank_model
        self.opt = optimizer
        self.scheduler = scheduler

        if fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams["training"]["num_workers"]

        if self.hparams["training"]["loss"] == "listnet":
            from micrank.losses.listNet import listNet
            self.loss = lambda x, y: listNet(x, y)
        elif self.hparams["training"]["loss"] == "topklistnet":
            from micrank.losses.listNet import AdaptiveTopKListNet
            self.loss = lambda x, y: AdaptiveTopKListNet(x, y)
        elif self.hparams["training"]["loss"] == "xentropy":
            self.loss = lambda x, y: (-(torch.log_softmax(x, -1)*y).sum() / x.shape[0])
        elif  self.hparams["training"]["loss"] == "mse":
            mseloss = torch.nn.MSELoss()
            self.loss = lambda x, y: mseloss(x, y)
        else:
            raise NotImplemented

        self.buffer_sel_train = []
        self.buffer_sel_val = []
        self.buffer_sel_test = []

        self.scaler = self._init_scaler()


    def extract_features(self, audio_tensor):

        if self.hparams["feats_type"] == "fbank":

            melspectra = torchaudio.transforms.MelSpectrogram(**self.hparams["fbank"]).to(audio_tensor)
            mels = melspectra(audio_tensor)
            amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="amplitude")
            logmels = amp_to_db(mels).clamp(min=-80, max=80)

            if self.hparams["augmentation"]["specaugm"] is None:
                return logmels

            batch, mics, mels, frames = logmels.shape
            logmels = logmels.reshape(batch*mics, mels, frames)

            if self.hparams["augmentation"]["specaugm"]["freqs"] is not None:
                mask_freq = torchaudio.transforms.FrequencyMasking(
                    self.hparams["augmentation"]["specaugm"]["freqs"], False)
                logmels = mask_freq(logmels)

            if self.hparams["augmentation"]["specaugm"]["time"] is not None:
                time_mask = torchaudio.transforms.FrequencyMasking(self.hparams["augmentation"]["specaugm"]["time"], False)
                logmels = time_mask(logmels)

            if self.hparams["augmentation"]["specaugm"]["shift"] is not None:
                for b in range(logmels.shape[0]):
                    for ch in range(logmels.shape[1]):
                        rolling = np.random.randint(0, logmels.shape[-1])
                        logmels[b, ch] = torch.roll(logmels[b, ch], rolling, dims=-1)
                #speed = torchaudio.transforms.TimeStretch(self.hparams["augmentation"]["specaugm"]["speed"], logmels.shape[1])
                #rate = np.random.uniform(0.95, 1.05)
                #logmels = speed(logmels, rate)

            return logmels.reshape(batch, mics, mels, frames)

        elif self.hparams["feats_type"] == "mels":
            melspectra = torchaudio.transforms.MelSpectrogram(**self.hparams["mels"]).to(audio_tensor)
            mels = melspectra(audio_tensor)
            return mels

        elif self.hparams["feats_type"] == "waveform":
            return audio_tensor

        elif self.hparams["feats_type"] == "spectra":
            return torchaudio.transforms.Spectrogram(audio_tensor)

        elif self.hparams["feats_type"] == "logspectra":
            spectra = torchaudio.transforms.Spectrogram(audio_tensor)
            amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="amplitude")
            return amp_to_db(spectra).clamp(min=-80, max=80)
        else:
            raise NotImplementedError

    def _init_scaler(self):
        """Scaler inizialization
        Raises:
            NotImplementedError: in case of not Implemented scaler
        Returns:
            TorchScaler: returns the scaler
        """

        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler("instance", self.hparams["scaler"]["normtype"], self.hparams["scaler"]["dims"])

            return scaler
        elif self.hparams["scaler"]["statistic"] == "dataset":
            # we fit the scaler
            scaler = TorchScaler(
                "dataset",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )
        else:
            raise NotImplementedError
        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"]
                    )
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader,
            transform_func=lambda x: self.extract_features(x.audio[0]))

        if self.hparams["scaler"]["savepath"] is not None:
            torch.save(scaler, self.hparams["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.hparams["scaler"]["savepath"]
                )
            )
            return scaler


    def training_step(self, batch, batch_indx):

        audio, lens = batch.audio
        scores, _ = batch.scores
        csid = batch.csid

        feats = self.scaler(self.extract_features(audio))

        if self.hparams["training"]["ranking"] in ["listwise", "pointwise"]:
            est = self.ranker(feats)
            loss = self.loss(est, scores)
            est_best = torch.argmax(est, -1)
        else:
            raise NotImplementedError

        with torch.no_grad():
            for b in range(audio.shape[0]):
                best_indx = est_best[b].item()
                self.buffer_sel_train.append([csid[b], batch.n_words[b].item(), batch.file_ids[b][best_indx]])

        self.log("train/loss", loss, prog_bar=True)

        return loss

    @staticmethod
    def compute_wer_all(buffer):

        csid, n_words, id = buffer[0]
        tot_csid = csid
        tot_n_words = n_words
        selected_ids = [id]
        for i in range(1, len(buffer)):
            c_csid, c_n_words, c_id = buffer[i]
            tot_csid = [tot_csid[j] + c_csid[j] for j in range(len(c_csid))]
            tot_n_words = tot_n_words + c_n_words
            selected_ids.append(c_id)

        return compute_wer(tot_csid[1], tot_csid[-1], tot_csid[-2], tot_n_words), selected_ids

    def validation_step(self, batch, batch_indx):

        audio, lens = batch.audio
        scores, _ = batch.scores
        csid = batch.csid

        feats = self.scaler(self.extract_features(audio))
        # for validation set we save all scores

        if self.hparams["training"]["ranking"] in ["listwise", "pointwise"]:
            est = self.ranker(feats)
            if not torch.any(torch.isnan(scores)):
                loss = self.loss(est, scores)
            else:
                loss = 0
                # chime6 has some utterances with 0 words we ignore them here
            #est_best = torch.argmax(est, -1)
        else:
            raise NotImplementedError

        for b in range(audio.shape[0]):

            score = est[b].item()
            self.buffer_sel_val.append([score, csid[b], batch.n_words[b].item(), batch.file_ids[b]])

        self.log("val/loss", loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self, **kwargs):

        # compute
        wer_best,  wer_top_3_avg = self.compute_wer_all(self.buffer_sel_val)

        if self.hparams["training"]["save_selected"]:
            out_dir = os.path.join(self.logger.log_dir, "dev")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "selected"), "w") as f:
                for c_sel in selected_val:
                    speaker = c_sel.split("_")[0]
                    start = c_sel.split(" ")[0].split("-")[1]
                    stop = c_sel.split(" ")[0].split("-")[-1]
                    session = c_sel.split(" ")[0].split("_")[1]
                    device = c_sel.split(" ")[0].split("_")[2].split(".")[0]
                    channel = c_sel.split(" ")[0].split("_")[2].split(".")[1]
                    f.write("{}_{}-{}-{} {}_{}.{} {} {}\n".format(speaker, session, start, stop, session, device, channel, int(start)/100, int(stop)/100))

        self.log("val/wer", wer_val, prog_bar=True)

        wer_train, _ = self.compute_wer_all(self.buffer_sel_train)

        self.log("train/wer", wer_train, prog_bar=True)

        self.buffer_sel_train = []
        self.buffer_sel_val = []



    def test_step(self, batch, batch_indx):

        audio, lens = batch.audio
        scores, _ = batch.scores
        csid = batch.csid

        feats = self.scaler(self.extract_features(audio))

        if self.hparams["training"]["ranking"] in ["listwise", "pointwise"]:
            est = self.ranker(feats)
            if not torch.any(torch.isnan(scores)):
                loss = self.loss(est, scores)
            else:
                loss = 0
            est_best = torch.argmax(est, -1)
        else:
            raise NotImplementedError

        for b in range(audio.shape[0]):
            best_indx = est_best[b].item()
            self.buffer_sel_test.append([csid[b][best_indx], batch.n_words[b].item(), batch.file_ids[b][best_indx]])

        self.log("test/loss", loss, prog_bar=True)

        return loss


    def test_step_end(self, *args, **kwargs):

        wer_test, selected_test = self.compute_wer_all(self.buffer_sel_test)

        if self.hparams["training"]["save_selected"]:

            out_dir = os.path.join(self.logger.log_dir, "eval")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "selected"), "w") as f:
                for c_sel in selected_test:
                    speaker = c_sel.split("_")[0]
                    start = c_sel.split(" ")[0].split("-")[1]
                    stop = c_sel.split(" ")[0].split("-")[-1]
                    session = c_sel.split(" ")[0].split("_")[1]
                    device = c_sel.split(" ")[0].split("_")[2].split(".")[0]
                    channel = c_sel.split(" ")[0].split("_")[2].split(".")[1]
                    f.write(
                        "{}_{}-{}-{} {}_{}.{} {} {}\n".format(speaker, session, start, stop, session, device, channel,
                                                              int(start) / 100, int(stop) / 100))

        self.log("test/wer", wer_test)
        self.log("hp_metric", wer_test)



    def configure_optimizers(self):
        return [self.opt], [self.scheduler]

    def train_dataloader(self):
        shuffle = True if self.hparams["training"]["sorted"] == "random" else False
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.hparams["training"]["batch_size"],
            num_workers=self.num_workers, shuffle=shuffle,
            collate_fn=PaddedBatch, worker_init_fn=lambda x: np.random.seed(
            int.from_bytes(os.urandom(4), "little") + x
        )

        )

        return self.train_loader

    def val_dataloader(self):
        self.val_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=PaddedBatch
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=PaddedBatch
        )
        return self.test_loader













