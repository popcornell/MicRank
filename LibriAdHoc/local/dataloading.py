import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
import torchaudio
import torch


def get_datasets(hparams):

    train_set = DynamicItemDataset.from_json(hparams["data"]["train_json"])
    dev_set = DynamicItemDataset.from_json(hparams["data"]["dev_json"])
    test_set = DynamicItemDataset.from_json(hparams["data"]["test_json"])
    if hparams["training"]["delta_wer"]:
        # we dump examples with wer difference less than X
        pass

    if hparams["training"]["sorted"]:
        # we filter based on words
        train_set = train_set.filtered_sorted(sort_key="length")

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("audio_file")
    @sb.utils.data_pipeline.provides("audio")
    def audio_pipeline(wav):
        sig, fs = torchaudio.load(wav)
        return sig

    if hparams["training"]["rankscore"] == "nwer":
        @sb.utils.data_pipeline.takes("channels", "n_words")
        @sb.utils.data_pipeline.provides("scores")
        def targets_pipeline(channels, n_words):
            targets = []
            for ch in channels:
                nwer = compute_nwer(ch["csid"], n_words)
                targets.append(nwer)

            return torch.Tensor(targets).float()
    else:
        raise NotImplementedError

    @sb.utils.data_pipeline.takes("channels")
    @sb.utils.data_pipeline.provides("csid")
    def csid_pipeline(channels):
        csid = []
        for ch in channels:
            csid.append(ch["csid"])
        return csid

    sb.dataio.dataset.add_dynamic_item([train_set, dev_set, test_set], audio_pipeline)
    sb.dataio.dataset.add_dynamic_item([train_set, dev_set], targets_pipeline)
    sb.dataio.dataset.add_dynamic_item([dev_set, test_set], csid_pipeline)

    sb.dataio.dataset.set_output_keys([train_set], ["id", "audio", "scores"])
    sb.dataio.dataset.set_output_keys([dev_set, test_set], ["id", "audio", "scores", "csid", "n_words"])

    return train_set, dev_set, test_set





