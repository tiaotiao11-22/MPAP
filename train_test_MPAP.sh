#cd /workspace/data1/Texture_Classification/Result/

export NCCL_SOCKET_IFNAME=eth0 
export GLOO_SOCKET_IFNAME=eth0 

read -p "Enter tag:" tag

#rm -rf *

cd /public/data1/users/zhaiwei16/GitCode/Texture_Classification

a="DTD"
N=10

if [ $a = "DTD" ];then
   N=10
elif [ $a = "FMD" ];then
   N=10
elif [ $a = "Car" ];then
   N=1
elif [ $a = "CUB" ];then
   N=1
elif [ $a = "GTOS" ];then
   N=5
elif [ $a = "GTOSM" ];then
   N=1
elif [ $a = "MINC" ];then
   N=5
else   #INDOOR
   N=1
fi

cd /public/data1/users/zhaiwei16/GitCode/Texture_Classification/Result

if [ ! -f "$filename" ]; then
   prefix=$a"_"
   suffix=$tag"_result.txt"
   filename=$prefix$suffix
   l=`sed -n '$=' $filename`
   if [ $l -gt 10 ] ; then
      rm -rf $filename
      l=1
   else
      l=`expr $l + 1`
   fi
else
   l=1
fi

cd /public/data1/users/zhaiwei16/GitCode/Texture_Classification

for i in $(seq $l $N)
do
   echo "Dataset: $a, Fold: $i, Begin: $l, End: $N, Tag: $tag"
   srun -A test -J J1 -N 1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 -p gpu -t 1-00:00:00 python MPAPNet_test.py --dataset $a --fold $i --tag $tag --batch_size 128 --accumulation_steps 1 --test_batch_size 8 --epochs 300 --labelsmoothing --a 0.5 --b 0.3 --c 0.2 --backbone resnet50 --lr 0.01
done

python Count.py --dataset $a --fold $N --tag $tag