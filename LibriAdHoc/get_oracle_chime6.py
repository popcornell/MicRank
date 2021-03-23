import json
from micrank.metrics import compute_wer
import numpy as np
import os

selection_method = "wer"
out_dir = "/tmp/RANDOM"
os.makedirs(out_dir, exist_ok=True)

for split in ["dev", "eval"]:

    with open(os.path.join("./parsed_chime6", split + ".json"), "r" )as f:
        utterances = json.load(f)


    selected_all = []
    for utt_id in utterances.keys():

        scores = []
        n_words = utterances[utt_id]["n_words"]
        for ch in range(len(utterances[utt_id]["channels"])):
            channels_data = utterances[utt_id]["channels"]
            c_csid = channels_data[ch]["csid"]
            c_id = channels_data[ch]["id"]

            if selection_method == "wer":
                try:
                    c_wer = compute_wer(c_csid[1], c_csid[-1], c_csid[-2], n_words)
                except ZeroDivisionError:
                    c_wer = 0
                scores.append(c_wer)

        if selection_method in ["wer"]:
            best_indx = np.argmin(scores)

        selected = utterances[utt_id]["channels"][best_indx]
        session = utterances[utt_id]["session"]
        start = utt_id.split(" ")[0].split("-")[1]
        stop = utt_id.split(" ")[0].split("-")[-1]

        selected_all.append("{} {} {} {}\n".format(utt_id, session + "_" + selected["channel"], int(start) / 100,
                                               int(stop) / 100))

    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    with open(os.path.join(out_dir, split,  "selected"), "w") as f:
        for s in selected_all:
            f.write(s)









