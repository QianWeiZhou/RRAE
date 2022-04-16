#37
set -o errexit
checkfolder="Your Folder/ck4-1-5/"

mkdir -p "${checkfolder}"
rm -rf $checkfolder/stop 
cp -r ../DRAW "${checkfolder}"

gpu="2"
dataset="mnist"
z_dim=4
multiplex=1
epochs=300
glimpses=5

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

checkfolder="Your Folder/ck4-2-5/"

mkdir -p "${checkfolder}"
rm -rf $checkfolder/stop 
cp -r ../DRAW "${checkfolder}"


dataset="mnist"
z_dim=4
multiplex=2
epochs=300
glimpses=5

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

checkfolder="Your Folder/ck4-3-5/"

mkdir -p "${checkfolder}"
rm -rf $checkfolder/stop 
cp -r ../DRAW "${checkfolder}"


dataset="mnist"
z_dim=4
multiplex=3
epochs=300
glimpses=5

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

checkfolder="Your Folder/ck4-4-5/"

mkdir -p "${checkfolder}"
rm -rf $checkfolder/stop 
cp -r ../DRAW "${checkfolder}"


dataset="mnist"
z_dim=4
multiplex=4
epochs=300
glimpses=5

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




