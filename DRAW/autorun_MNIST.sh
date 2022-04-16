#37
set -o errexit

checkfolder="Your Folder/ck1-1-2/"

mkdir -p "${checkfolder}"
rm -rf $checkfolder/stop 
cp -r ../DRAW "${checkfolder}"

gpu="1"
dataset="mnist"
z_dim=1
multiplex=1
epochs=300
glimpses=2

echo "${checkfolder}"
echo "${epochs}"

if true; then
python train.py -ep ${epochs} \
                        --multiplex ${multiplex} \
                        -z ${z_dim} \
                        --dataset ${dataset} \
                        --gpu ${gpu} \
                        -cf ${checkfolder}\
                        -g ${glimpses}
                  
fi

checkfolder="Your Folder/ck1-2-2/"

mkdir -p "${checkfolder}"
rm -rf $checkfolder/stop 
cp -r ../DRAW "${checkfolder}"


dataset="mnist"
z_dim=1
multiplex=2
epochs=300
glimpses=2

echo "${checkfolder}"
echo "${epochs}"

if true; then
python train.py -ep ${epochs} \
                        --multiplex ${multiplex} \
                        -z ${z_dim} \
                        --dataset ${dataset} \
                        --gpu ${gpu} \
                        -cf ${checkfolder}\
                        -g ${glimpses}
                  
fi

checkfolder="Your Folder/ck1-3-2/"

mkdir -p "${checkfolder}"
rm -rf $checkfolder/stop 
cp -r ../DRAW "${checkfolder}"


dataset="mnist"
z_dim=1
multiplex=3
epochs=300
glimpses=2

echo "${checkfolder}"
echo "${epochs}"

if true; then
python train.py -ep ${epochs} \
                        --multiplex ${multiplex} \
                        -z ${z_dim} \
                        --dataset ${dataset} \
                        --gpu ${gpu} \
                        -cf ${checkfolder}\
                        -g ${glimpses}
                  
fi

checkfolder="Your Folder/ck1-4-2/"

mkdir -p "${checkfolder}"
rm -rf $checkfolder/stop 
cp -r ../DRAW "${checkfolder}"


dataset="mnist"
z_dim=1
multiplex=4
epochs=300
glimpses=2

echo "${checkfolder}"
echo "${epochs}"

if true; then
python train.py -ep ${epochs} \
                        --multiplex ${multiplex} \
                        -z ${z_dim} \
                        --dataset ${dataset} \
                        --gpu ${gpu} \
                        -cf ${checkfolder}\
                        -g ${glimpses}
                  
fi


#mv "${checkfolder}""modelfolder" "${archive_dir}"
#cp $BASH_SOURCE "${archive_dir}"




