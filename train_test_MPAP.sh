export NCCL_SOCKET_IFNAME=eth0 
export GLOO_SOCKET_IFNAME=eth0 

read -p "Enter Dataset Name: " a
tag=1

cd /home/weizhai/GitCode/Texture_Classification      #your project root

if [ $a = "DTD" ];then
   N=10
elif [ $a = "FMD" ];then
   N=10
elif [ $a = "KTH" ];then   # run 10
   N=4
elif [ $a = "GTOS" ];then
   N=5
else                       #GTOSM run 2
   N=1
fi

cd /home/weizhai/GitCode/Texture_Classification/Result
rm -rf *
cd /home/weizhai/GitCode/Texture_Classification

B=1

for i in $(seq $B $N)
do
   echo "Dataset: $a, Fold: $i, Begin: $B, End: $N, Tag: $tag"
   python MPAPNet_test.py --dataset $a --fold $i --tag $tag --batch_size 128 --accumulation_steps 1 --test_batch_size 8 --epochs 2 --labelsmoothing --a 0.5 --b 0.3 --c 0.2 --backbone resnet50 --lr 0.01
done

python Count.py --dataset $a --fold $N --tag $tag