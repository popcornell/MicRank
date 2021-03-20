libriadhoc_root=/media/samco/639f877e-4513-4484-91e2-1659d975147d/micrank_synth


for split in train dev test; do
  echo Parsing ${split} set
  python local/parse_data.py --per_utt $libriadhoc_root/wer_details_${split}_tgsmall/per_utt   \
  --audio_dir ${libriadhoc_root}/$split/noisy \
  --meta_dir ${libriadhoc_root}/$split/meta \
   --json_file ./parsed/${split}.json
done
