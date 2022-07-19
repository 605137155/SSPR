export PYTHONPATH=$PYTHONPATH:./
python ./example/iics_threshold.py --dataset market1501 --checkpoint /content/drive/MyDrive/zmh/C/checkpoint/resnet50-19c8e357.pth \
--data-dir /content/sample_data/data -j 0
