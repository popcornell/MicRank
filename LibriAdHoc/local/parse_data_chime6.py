import argparse
import os
import json
from pathlib import Path
import tqdm
import soundfile as sf

parser = argparse.ArgumentParser("parsing wer and wac from per_utt kaldi files")
parser.add_argument("--per_utt", default="/media/samco/639f877e-4513-4484-91e2-1659d975147d/ChannelSelect/egs/labels/per_utt_train")
parser.add_argument("--audio_dir", default="/media/samco/639f877e-4513-4484-91e2-1659d975147d/CHiME6/audio/train")
#parser.add_argument("--meta_dir", default="/media/sam/bx500/micrank_synth/train/meta")
parser.add_argument("--json_file", default="./parsed_chime6/train.json")


def parse_examples(per_utt_file, audio_dir, json_file):

    with open(per_utt_file, "r") as f:
        scores = f.readlines()

    labels = {}
    for indx in tqdm.tqdm(range(len(scores))):
        l = scores[indx]
        if l.split(" ")[1] == "ref":
            n_words = len([x for x in l.strip("\n").split(" ")[2:] if x != ""])
        if l.split(" ")[1] == "#csid":
            csid = [int(x) for x in l.strip("\n").split(" ")[2:]]

            start = l.split(" ")[0].split("-")[1]
            stop = l.split(" ")[0].split("-")[-1]
            session = l.split(" ")[0].split("_")[1]
            speaker = l.split(" ")[0].split("_")[0]
            device = l.split(" ")[0].split("_")[2].split(".")[0]
            channel = l.split(" ")[0].split("_")[2].split(".")[1]
            tag = l.split(" ")[0].split("_")[2].split(".")[2].split("-")[0]

            if channel in ["L", "R"] or session in ["S24", "S19", "S03"]:
                continue
            audio_file = os.path.join(audio_dir, session + "_" + device + "." + channel + ".wav")
            os.path.isfile(audio_file)

            utt_id = speaker + "_" + session + "-" + start + "-" + stop


            if utt_id not in labels.keys():
                labels[utt_id] = {"n_words": n_words,
                                  "length": (int(stop) - int(start))*160,
                                  "session": session,
                                  "tag": tag,
                                   "s_start": int(start)*160, "s_stop": int(stop)*160,
                                  "speaker": speaker,
                                  "channels": [{"audio_file": audio_file, "channel": device + "." + channel, "csid": csid, "id": l.split(" ")[0]}]}

            else:
                labels[utt_id]["channels"].append({"audio_file": audio_file, "channel": device + "." + channel, "csid": csid, "id": l.split(" ")[0]})

    os.makedirs(Path(json_file).parent, exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(labels, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    parse_examples(args.per_utt, args.audio_dir, args.json_file)