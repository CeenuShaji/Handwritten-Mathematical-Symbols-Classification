#!/bin/bash
#SBATCH --job-name=EEL5840_YOLO
#SBATCH --output=RETRAIN_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=justin.rossiter@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --account=eel5840
#SBATCH --qos=eel5840

python detect.py --img 320 --weights runs/train/exp13/weights/last.pt --source /blue/eel5840/justin.rossiter/test/images --data eel5840.yaml

date
