cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

STEP='best'

CONFIG_PATH='config_release/ovqa.json'

TXT_DB='data\ovqa_annos\ovqa_test.jsonl'
IMG_DB='/temp/' # TODO

horovodrun -np 1 python src/tasks/run_ovqa.py  \
      --do_inference 1 \
      --inference_split test \
      --inference_model_step $STEP \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 32 \
      --output_dir output/ovqa/20220922_test \
      --config $CONFIG_PATH