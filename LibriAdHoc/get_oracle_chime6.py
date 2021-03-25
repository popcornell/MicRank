import json
from micrank.metrics import compute_wer
import os

selection_method = "wer"
out_dir = "/tmp/WER"
os.makedirs(out_dir, exist_ok=True)

for split in ["dev", "eval"]:

    with open(os.path.join("./parsed_chime6", split + ".json"), "r" )as f:
        utterances = json.load(f)

    ALL = []
    for utt_id in utterances.keys():

        current_utt = []
        n_words = utterances[utt_id]["n_words"]
        for ch in range(len(utterances[utt_id]["channels"])):
            channels_data = utterances[utt_id]["channels"]
            c_csid = channels_data[ch]["csid"]
            c_id = channels_data[ch]["id"]

            if selection_method == "wer":
                try:
                    score = compute_wer(c_csid[1], c_csid[-1], c_csid[-2], n_words)
                except ZeroDivisionError:
                    score = 0
                current_utt.append([score, c_csid, c_id])

        ALL.append(current_utt)

    with open(os.path.join(out_dir, "scores_" + split + "_" + selection_method + ".json"), "w") as f:
        json.dump(ALL, f, indent=4)

    # now we sort the scores and then write the top 3 selection files

    SORTED_ALL = []  # we keep a list for all sorted
    for indx in range(len(ALL)):
        if selection_method in ["wer", "distance_spk", "rev_noise_energy", "cs_informed", "entropy_dnn"]:
                sorted_scores = sorted(ALL[indx], key=lambda x: x[0])
        elif selection_method in ["sisdr_clean", "sisdr_rev", "sisdr_clean_a",
                                    "sdr_clean", "sdr_rev",
                                    "pesq_clean_a", "pesq_clean", "pesq_rev",
                                    "stoi_clean_a", "stoi_clean", "stoi_rev",
                                    "distance_noise",
                                    "random",
                                    "rev_spk_energy"]:
                sorted_scores = sorted(ALL[indx], key=lambda x: x[0], reverse=True)
        else:
            raise NotImplementedError
        SORTED_ALL.append(sorted_scores)

    for top in range(3):
        os.makedirs(os.path.join(out_dir, "{}_{}".format(selection_method, top)), exist_ok=True)
        with open(os.path.join(out_dir, "{}_{}".format(selection_method, top), "selected"), "w") as f:
            for indx in range(len((SORTED_ALL))):
                c_id = SORTED_ALL[indx][top][-1]
                session = c_id.split("_")[1]
                start = c_id.split("-")[1]
                stop = c_id.split("-")[2].split(" ")[0]
                array = c_id.split(".")[0].split("_")[-1]
                channel = c_id.split(".")[1]

                f.write("{} {} {} {}\n".format(c_id, session + "_" + array + "." + channel, int(start) / 100, int(stop) / 100))