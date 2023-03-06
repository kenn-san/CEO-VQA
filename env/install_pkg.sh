apt update
apt install lsof

# horovod
HOROVOD_GPU_ALLREDUCE=NCCL  HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod[pytorch] && ldconfig

# use the faster pillow-simd instead of the original pillow
# https://github.com/uploadcare/pillow-simd
pip uninstall pillow && \
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

spacy download en

pip install -r requirements.txt

# NVIDIA apex
git clone https://github.com/NVIDIA/apex.git &&\
cd apex &&\
    python setup.py install &&\
    rm -rf ../apex

