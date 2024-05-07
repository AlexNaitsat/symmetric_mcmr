# input params: <dataset_path> <classes> <modes> <sample_sufix> <sample_ratio>
#
#Examples:
# 1) Sample  1/10 of "aeroplane_train.txt" into "aeroplane_small_train.txt": 
#  mcmr> bash scripts/sample_from_index_files.sh   /home/ubuntu/research/datasets/Pascal3D aeroplane train small 10
#        head -190 /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_train.txt > /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_small_train.txt

# 2) Sample 1/10 from 4 index files:
#  mcmr> bash scripts/sample_from_index_files.sh   /home/ubuntu/research/datasets/Pascal3D "bus aeroplane" "test train" small 10
#        head -108 /home/ubuntu/research/datasets/Pascal3D/bus/bus_train.txt > /home/ubuntu/research/datasets/Pascal3D/bus aeroplane/bus_small_train.txt
#        head -15  /home/ubuntu/research/datasets/Pascal3D/bus/bus_test.txt > /home/ubuntu/research/datasets/Pascal3D/bus aeroplane/bus_small_test.txt
#        head -190 /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_train.txt > /home/ubuntu/research/datasets/Pascal3D/bus aeroplane/aeroplane_small_train.txt
#        head -29  /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_test.txt > /home/ubuntu/research/datasets/Pascal3D/bus aeroplane/aeroplane_small_test.txt


#for multiple index files 
class_list="$2"
mode_list="$3"
min_sample_num="$6"
for class in $class_list;
do
for mode in $mode_list;
    do
        in_index_file="$1/${class}/${class}_${mode}.txt"
        out_index_file="$1/${class}/${class}_${4}_${mode}.txt"

        #echo $test_index_file
        all_sample_num=($(wc  -l $in_index_file))
        #echo $all_sample_num
        let "sample_num = $all_sample_num / $5"
        sample_num=$((sample_num <  min_sample_num  ? min_sample_num: sample_num ))
        #echo $sample_num

        echo "head -${sample_num} $in_index_file > $out_index_file"
        head -${sample_num} $in_index_file > $out_index_file
    done
done 

# #for single input index file 
# in_index_file="$1/${2}/${2}_${3}.txt"
# out_index_file="$1/${2}/${2}_${4}_${3}.txt"
# all_sample_num=($(wc  -l $in_index_file))
# let "sample_num = $all_sample_num / $5"
# echo "head -${sample_num} $in_index_file > $out_index_file"
# head -${sample_num} $in_index_file > $out_index_file


