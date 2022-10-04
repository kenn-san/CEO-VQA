cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH


CONFIG_PATH='config_release/ovqa.json'

TXT_DB='data/ovqa_annos/ovqa_test.jsonl'
IMG_DB='/temp/generated_4fps' # TODO

for (( STEP=1800; STEP>=600; STEP-=300 ))
do 
horovodrun -np 1 python src/tasks/run_ovqa.py  \
      --do_inference 1 \
      --inference_split test \
      --inference_model_step $STEP \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 32 \
      --output_dir output/ovqa/20220927_4fps_test \
      --config $CONFIG_PATH
done