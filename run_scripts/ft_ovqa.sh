cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/ovqa.json'

horovodrun -np 1 python src/tasks/run_ovqa.py \
      --config $CONFIG_PATH \
      --output_dir output/ovqa/20220925_test_in_model_mcq #\
      #--debug 1
      
