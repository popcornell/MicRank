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

parser = argparse.ArgumentParser("Getting Oracle Ranking")
parser.add_argument("--json_file")



def get_signal_based_metric(audio_file, metadata_channel, metric, align=False, rev_ref=True):
    noisy, _ = sf.read(audio_file)
    noisy = noisy[:, metadata_channel["channel"]]

    if rev_ref:
        name = Path(audio_file).stem
        reference, _ = sf.read(os.path.join(Path(audio_file).parent.parent, "clean", name + ".wav"))
        reference = reference[:, metadata_channel["channel"]]


    metrics = get_metrics(noisy, reference, noisy, 16000, metrics_list=[metric])
    return metrics[metric]




def rank(json_file, selection_method):

    with open(json_file, "r") as f:
        utterances = json.load(f)

    min_csid = []
    for id in tqdm.tqdm(utterances.keys()):

        n_words = utterances[id]["n_words"]
        scores = []
        for ch in range(len(utterances[id]["channels"])):
            channels_data = utterances[id]["channels"]
            c_csid = channels_data[ch]["csid"]
            c_id = channels_data[ch]["id"]

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
                pass
            elif selection_method == "pesq_clean":
                pass
            elif selection_method == "pesq_rev":
                pesq_rev = get_signal_based_metric(utterances[id]["audio_file"], channels_data[ch], "pesq", False, True)
                scores.append(pesq_rev)
            elif selection_method == "stoi_rev":
                stoi_rev = get_signal_based_metric(utterances[id]["audio_file"], channels_data[ch], "stoi", False, True)
                scores.append(stoi_rev)
            elif selection_method == "sisdr_rev":
                sisdr = get_signal_based_metric(utterances[id]["audio_file"], channels_data[ch], "sisdr", False, True)
                scores.append(sisdr)
            elif selection_method == "sdr_rev":
                sdr = get_signal_based_metric(utterances[id]["audio_file"], channels_data[ch], "sdr_rev", False, True)
                scores.append(sdr)

        if selection_method in ["wer", "distance_spk", "rev_noise_energy"]:
            best_indx = np.argmin(scores)
        elif selection_method in ["sisdr_clean", "sisdr_rev",
                                  "sdr_clean", "sdr_rev",
                                  "pesq_clean_a", "pesq_clean", "pesq_rev",
                                  "stoi_clean_a", "stoi_clean", "stoi_rev",
                                  "distance_noise",
                                  "random",
                                  "rev_spk_energy"]:
            best_indx = np.argmax(scores)

        min_csid.append([channels_data[best_indx]["csid"], n_words, c_id])

    oracle_wer, selected = MicRank.compute_wer_all(min_csid)
    print("Oracle WER {} file {} method {}", oracle_wer, json_file, selection_method)

if __name__ == "__main__":
    np.random.seed(42)
    rank("/media/samco/Data/MicRank/LibriAdHoc/parsed/test.json", "stoi_rev")


