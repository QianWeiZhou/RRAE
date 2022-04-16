set -o errexit
#source activate pytorch
checkfolder="Your Folder"
mkdir -p "${checkfolder}"
cp -r ../VAE-CVAE "${checkfolder}"

gpu="2"

python train.py --epochs 300 --latent_size 2 -m 1 --conditional --checkfolder ${checkfolder} --gpu ${gpu}
python train.py --epochs 300 --latent_size 2 -m 2 --conditional --checkfolder ${checkfolder} --gpu ${gpu}
python train.py --epochs 300 --latent_size 2 -m 3 --conditional --checkfolder ${checkfolder} --gpu ${gpu}
python train.py --epochs 300 --latent_size 2 -m 4 --conditional --checkfolder ${checkfolder} --gpu ${gpu}

python train.py --epochs 300 --latent_size 5 -m 1 --conditional --checkfolder ${checkfolder} --gpu ${gpu}
python train.py --epochs 300 --latent_size 5 -m 2 --conditional --checkfolder ${checkfolder} --gpu ${gpu}
python train.py --epochs 300 --latent_size 5 -m 3 --conditional --checkfolder ${checkfolder} --gpu ${gpu}
python train.py --epochs 300 --latent_size 5 -m 4 --conditional --checkfolder ${checkfolder} --gpu ${gpu}

python train.py --epochs 300 --latent_size 10 -m 1 --conditional --checkfolder ${checkfolder} --gpu ${gpu}
python train.py --epochs 300 --latent_size 10 -m 2 --conditional --checkfolder ${checkfolder} --gpu ${gpu}
python train.py --epochs 300 --latent_size 10 -m 3 --conditional --checkfolder ${checkfolder} --gpu ${gpu}
python train.py --epochs 300 --latent_size 10 -m 4 --conditional --checkfolder ${checkfolder} --gpu ${gpu}

python train.py --epochs 300 --latent_size 20 -m 1 --conditional --checkfolder ${checkfolder} --gpu ${gpu}
python train.py --epochs 300 --latent_size 20 -m 2 --conditional --checkfolder ${checkfolder} --gpu ${gpu}
python train.py --epochs 300 --latent_size 20 -m 3 --conditional --checkfolder ${checkfolder} --gpu ${gpu}
python train.py --epochs 300 --latent_size 20 -m 4 --conditional --checkfolder ${checkfolder} --gpu ${gpu}