export PYTHONPATH=$PYTHONPATH:./
python ./example/iics.py --dataset market1501 --checkpoint /content/drive/MyDrive/zmh/little/checkpoint/resnet50-19c8e357.pth \ --evaluate 
--data-dir /content/sample_data/data  --cluster_epochs 40
