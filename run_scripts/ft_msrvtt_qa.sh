cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/msrvtt_qa.json'

horovodrun -np 1 python src/tasks/run_video_qa.py \
      --config $CONFIG_PATH \
      --output_dir output/ovqa/20221024_offline_0.4_7_backward_9_forward_fib_sample