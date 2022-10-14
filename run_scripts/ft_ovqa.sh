cd .. 

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/ovqa.json'

horovodrun -np 1 python src/tasks/run_ovqa.py \
      --config $CONFIG_PATH \
      --output_dir output/ovqa/20221011_1_by_1_test #\
      #--debug 1
      
