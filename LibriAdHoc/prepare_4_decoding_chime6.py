import numpy as np
import os
from tqdm import tqdm

in_dir_selected = "/media/samco/Data/MicRank/LibriAdHoc/exp_chime6/cd_corrected/version_1"
baseline_dir = "/media/samco/Data/MicRank/baseline_beamformIT"


for split in ["dev", "eval"]:

    with open(os.path.join(baseline_dir, split, "segments"), "r") as f:
        baseline_segs = f.readlines()

    with open(os.path.join(in_dir_selected, split, "selected"), "r") as f:
        selected_segs = f.readlines()

    # create an hash table with selected
    hashsel = {}
    for l in selected_segs:
        speaker = l.split("_")[0]
        session = l.split("_")[1]
        selected = l.split(" ")[1]
        start = l.split(" ")[2]
        stop = l.split(" ")[-1]
        utt_id = "{}_{}_{}_{}".format(speaker, session, start, stop)
        hashsel[utt_id] = selected

    out = []
    for l in tqdm(baseline_segs):
        speaker = l.split("_")[0]
        session = l.split("_")[1]
        selected = l.split(" ")[1]
        place = l.split(".ENH")[0].split("_")[-1]
        start = l.split(" ")[2]
        stop = l.split(" ")[-1]
        #utt_id = "{}_{}_{}_{}\n".format(speaker, session, float(start), float(stop))

        # find corresponding key
        for k in hashsel.keys():
            k_start = float(k.split("_")[2])
            k_stop = float(k.split("_")[-1])

            if abs(float(start) - k_start) <= 0.01 or abs(float(stop) - k_stop) <= 0.01:
                break

        selected_channel = hashsel[k]
        array, channel = selected_channel.split("_")[-1].split(".")

        out.append("{}_{}_{}_{}.{}-{}-{} {} {} {}".format(speaker, session, array, place, channel, l.split("-")[1], l.split("-")[2].split(" ")[0], selected_channel, start, stop))


    out = sorted(out, key=lambda x: x.split(" ")[0])
    with open(os.path.join(in_dir_selected, split, "segments"), "w") as f:
        f.writelines(out)


