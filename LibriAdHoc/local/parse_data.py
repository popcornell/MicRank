import argparse
import os
import json
from pathlib import Path
import tqdm
import soundfile as sf

parser = argparse.ArgumentParser("parsing wer and wac from per_utt kaldi files")
parser.add_argument("--per_utt", default="/media/sam/bx500/micrank_synth/wer_details_train_tgsmall/per_utt")
parser.add_argument("--audio_dir", default="/media/sam/bx500/micrank_synth/train/noisy")
parser.add_argument("--meta_dir", default="/media/sam/bx500/micrank_synth/train/meta")
parser.add_argument("--json_file", default="./parsed/train.json")


def parse_examples(per_utt_file, audio_dir, metadata_dir, json_file):

    with open(per_utt_file, "r") as f:
        scores = f.readlines()

    labels = {}
    for indx in tqdm.tqdm(range(len(scores))):
        l = scores[indx]
        if l.split(" ")[1] == "ref":
            n_words = len([x for x in l.strip("\n").split(" ")[2:] if x != ""])
        if l.split(" ")[1] == "#csid":
            csid = [int(x) for x in l.strip("\n").split(" ")[2:]]
            utt_id = l.split("-CH")[0]
            channel= int(l.split("-CH")[1].split(" ")[0])

            audio_file = os.path.join(audio_dir, utt_id + ".wav")
            os.path.isfile(audio_file)

            with open(os.path.join(metadata_dir, utt_id + ".json"), "r") as f:
                metadata = json.load(f)

            c_metadata = metadata[utt_id + "--CH" + str(channel)]

            if utt_id not in labels.keys():
                labels[utt_id] = {"audio_file": audio_file,
                                  "n_words": n_words,
                                  "length": len(sf.SoundFile(audio_file)),
                                  "channels": [{"channel": channel, "csid": csid, "id": "dummy",  **c_metadata}]}

            else:
                labels[utt_id]["channels"].append({"channel": channel, "csid": csid, "id": "dummy",  **c_metadata})

    os.makedirs(Path(json_file).parent, exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(labels, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    parse_examples(args.per_utt, args.audio_dir, args.meta_dir, args.json_file)