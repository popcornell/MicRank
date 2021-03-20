chime6_audio_root=/media/samco/639f877e-4513-4484-91e2-1659d975147d/CHiME6/audio/
per_utt_dir=/media/samco/639f877e-4513-4484-91e2-1659d975147d/ChannelSelect/egs/labels/

for split in train dev eval; do
  echo Parsing ${split} set
  python local/parse_data_chime6.py --per_utt ${per_utt_dir}/per_utt_${split}   \
  --audio_dir ${chime6_audio_root}/$split/ \
   --json_file ./parsed_chime6/${split}.json
done
