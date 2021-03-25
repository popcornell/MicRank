import argparse
import os
import json
from micrank.metrics import compute_wer
import tqdm
import numpy as np
from local.rank_trainer import MicRank
from asteroid.metrics import get_metrics
import soundfile as sf
from pathlib import Path
from scipy.ndimage import shift
from micrank.rankers.dsp_selection import CepstralDistance
import torch
from local.align import phase_align
import pandas as pd

parser = argparse.ArgumentParser("Getting Oracle Ranking")
parser.add_argument("--json_file")



from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech",
                                               savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")

def get_signal_based_metric(audio_file, librispeech, metadata_channel, metric, align=False, rev_ref=True):
    noisy, _ = sf.read(audio_file)
    noisy = noisy[:, metadata_channel["channel"]]

    if rev_ref:
        name = Path(audio_file).stem
        reference, _ = sf.read(os.path.join(Path(audio_file).parent.parent, "clean", name + ".wav"))
        reference = reference[:, metadata_channel["channel"]]
    else:
        orig_speech = os.path.join(librispeech, "/".join(metadata_channel["orig_speech"].split("/")[-4:]))
        reference, _ = sf.read(orig_speech)
        noisy = noisy[:len(reference)]


    if align:
        n_samples = int(((metadata_channel["dist_mic_spk"]/340)/(1/16000)))
        #name_clean_rev = Path(audio_file).stem
        #reference_clean_rev, _ = sf.read(os.path.join(Path(audio_file).parent.parent, "clean", name_clean_rev + ".wav"))
        #reference_clean_rev = reference_clean_rev[:, metadata_channel["channel"]]
        #shift_sample = int(phase_align(reference_clean_rev, reference , (0, 400)))
        reference= shift(reference, n_samples)


    if metric == "cs_informed":
        cs = CepstralDistance()
        score =cs.from_oracle(torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float(), torch.from_numpy(reference).unsqueeze(0).float())
        score = score.item()
        return score

    elif metric == "entropy_dnn":
        with torch.no_grad():
            wav_lens = torch.tensor([1.])
            encoder_out = asr_model.encode_batch(torch.from_numpy(noisy).unsqueeze(0), wav_lens=wav_lens)
            _, scores = asr_model.modules.beam_searcher(
                encoder_out, wav_lens
            )

        entropy = -torch.sum(scores*torch.log2(scores), -1)
        entropy = entropy.mean()
        return entropy

    else:

        metrics = get_metrics(noisy, reference, noisy, 16000, metrics_list=[metric])

    return metrics[metric]


def rank(json_file, librispeech_dir, selection_method, out_file):

    with open(json_file, "r") as f:
        utterances = json.load(f)

    dataframe = []

    for id in tqdm.tqdm(utterances.keys()):


        n_words = utterances[id]["n_words"]
        scores = []
        CSIDs = []
        for ch in range(len(utterances[id]["channels"])):
            channels_data = utterances[id]["channels"]
            c_csid = channels_data[ch]["csid"]

            CSIDs.append(c_csid)

            if selection_method == "wer":
                c_wer = compute_wer(c_csid[1], c_csid[-1], c_csid[-2], n_words)
                scores.append(c_wer)
            elif selection_method == "distance_spk":
                c_dist = channels_data[ch]["dist_mic_spk"]
                scores.append(c_dist)
            elif selection_method == "distance_noise":
                c_dist = channels_data[ch]["dist_mic_noise"]
                scores.append(c_dist)
            elif selection_method == "random":
                ran = np.random.rand()
                scores.append(ran)
            elif selection_method == "rev_spk_energy":
                c_dist = channels_data[ch]["rev_speech_dB"]
                scores.append(c_dist)
            elif selection_method == "rev_noise_energy":
                c_dist = channels_data[ch]["rev_noise_dB"]
                scores.append(c_dist)
            elif selection_method == "pesq_clean_a":
                pesq_clean = get_signal_based_metric(utterances[id]["audio_file"],
                                                     librispeech_dir,
                                                     channels_data[ch], "pesq", True, False)
                scores.append(pesq_clean)
            elif selection_method == "pesq_clean":
                pesq_clean = get_signal_based_metric(utterances[id]["audio_file"],
                                                   librispeech_dir,
                                                   channels_data[ch], "pesq", False, False)
                scores.append(pesq_clean)
            elif selection_method == "pesq_rev":
                pesq_rev = get_signal_based_metric(utterances[id]["audio_file"],
                                                   librispeech_dir,
                                                   channels_data[ch], "pesq", False, True)
                scores.append(pesq_rev)
            elif selection_method == "stoi_rev":
                stoi_rev = get_signal_based_metric(utterances[id]["audio_file"],
                                                   librispeech_dir,
                                                   channels_data[ch], "stoi", False, True)
                scores.append(stoi_rev)
            elif selection_method == "stoi_clean":
                stoi_rev = get_signal_based_metric(utterances[id]["audio_file"],
                                                   librispeech_dir,
                                                   channels_data[ch], "stoi", False, False)
                scores.append(stoi_rev)
            elif selection_method == "stoi_clean_a":
                stoi_rev = get_signal_based_metric(utterances[id]["audio_file"],
                                                   librispeech_dir,
                                                   channels_data[ch], "stoi", True, False)
                scores.append(stoi_rev)
            elif selection_method == "sisdr_rev":
                sisdr = get_signal_based_metric(utterances[id]["audio_file"],
                                                librispeech_dir,
                                                channels_data[ch], "si_sdr", False, True)
                scores.append(sisdr)
            elif selection_method == "sisdr_clean_a":
                sisdr = get_signal_based_metric(utterances[id]["audio_file"],
                                                librispeech_dir,
                                                channels_data[ch], "si_sdr", True, False)
                scores.append(sisdr)
            elif selection_method == "sdr_rev":
                sdr = get_signal_based_metric(utterances[id]["audio_file"],
                                              librispeech_dir,
                                              channels_data[ch], "sdr", False, True)
                scores.append(sdr)
            elif selection_method == "sdr_clean":
                sdr = get_signal_based_metric(utterances[id]["audio_file"],
                                              librispeech_dir,
                                              channels_data[ch], "sdr", False, False)
                scores.append(sdr)

            elif selection_method == "cs_informed":
                sdr = get_signal_based_metric(utterances[id]["audio_file"],
                                              librispeech_dir,
                                              channels_data[ch], "cs_informed", True, False)
                scores.append(sdr)
            elif selection_method == "entropy_dnn":

                entropy = get_signal_based_metric(utterances[id]["audio_file"],
                                              librispeech_dir,
                                              channels_data[ch], "entropy_dnn", False, False)
                scores.append(entropy)


        #if selection_method in ["wer", "distance_spk", "rev_noise_energy", "cs_informed", "entropy_dnn"]:
        #    best_indx = np.argmin(scores)
        #elif selection_method in ["sisdr_clean", "sisdr_rev", "sisdr_clean_a",
         #                         "sdr_clean", "sdr_rev",
         #                         "pesq_clean_a", "pesq_clean", "pesq_rev",
         #                         "stoi_clean_a", "stoi_clean", "stoi_rev",
         #                         "distance_noise",
         #                         "random",
          #                        "rev_spk_energy"]:
          #  best_indx = np.argmax(scores)
        current = {"scores": [(x, y) for x, y in zip(scores, CSIDs)]}
        dataframe.append(current)

    with open(out_file, "w") as f:
        json.dump(dataframe, f, indent=4)
    #oracle_wer, selected = MicRank.compute_wer_all(min_csid)


if __name__ == "__main__":
    np.random.seed(42)

    json_utterances = "./parsed/test.json"
    librispeech_root = "/media/samco/Data/LibriSpeech/"

    out_dir = "./oracle_libriadhoc_test"
    os.makedirs(out_dir, exist_ok=True)

    #"wer", "distance_spk", "rev_noise_energy", "cs_informed", "sisdr_clean", "sisdr_rev", "sisdr_clean_a",
                             #    "sdr_clean", "sdr_rev",
                              #    "pesq_clean_a",

    for ranking_metric in [# "wer", "distance_spk", "rev_noise_energy", "cs_informed", "sisdr_clean", "sisdr_rev", "sisdr_clean_a",
                                 "sdr_clean", "sdr_rev",
                                  "pesq_clean_a", "pesq_clean", "pesq_rev",
                                  "stoi_clean_a", "stoi_clean", "stoi_rev",
                                  "distance_noise",
                                  "random",
                                  "rev_spk_energy"]:
        out_file = os.path.join(out_dir, ranking_metric + ".json")
        rank(json_utterances, librispeech_root, ranking_metric, out_file)
        with open(out_file, "r") as f:
            scores = json.load(f)

        all_csid = [] # we keep a list for all csids

        for f in range(len(scores)):
            if ranking_metric in ["wer", "distance_spk", "rev_noise_energy", "cs_informed", "entropy_dnn"]:
                sorted_scores = sorted(scores[f]["scores"], key=lambda x: x[0])
            elif ranking_metric in ["sisdr_clean", "sisdr_rev", "sisdr_clean_a",
                                 "sdr_clean", "sdr_rev",
                                  "pesq_clean_a", "pesq_clean", "pesq_rev",
                                  "stoi_clean_a", "stoi_clean", "stoi_rev",
                                  "distance_noise",
                                  "random",
                                  "rev_spk_energy"]:
                sorted_scores = sorted(scores[f]["scores"], key=lambda x: x[0], reverse=True)
            else:
                raise NotImplementedError
            all_csid.append([x[-1] for x in sorted_scores])

        # compute wer for all
        # print wer to txt file
        all_csid = np.sum(np.array(all_csid), 0)
        wers = [compute_wer(all_csid[r][1],all_csid[r][-1], all_csid[r][-2], np.sum(all_csid[r])) for r in range(len(all_csid))]
        with open(os.path.join(out_dir, ranking_metric + "_tot_wers.json"), "w") as f:
            json.dump(wers, f, indent=4)






