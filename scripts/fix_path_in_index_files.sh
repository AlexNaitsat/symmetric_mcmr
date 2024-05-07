# Input params: <dataset_path> <classes> <modes> 
# it  fixes dataset path of yaml annotation files inside gicen index files  
#Examples:
# 1) Fix one trian index file 
#  mcmr>  bash scripts/fix_path_in_index_files.sh /home/ubuntu/research/datasets/Pascal3D aeroplane  train 
#         cp /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_train.txt /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_train.txt.backup
#         awk -v p=/home/ubuntu/research/datasets/Pascal3D/aeroplane/annotations/  -F'/' '{print p $NF}' /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_train.txt > /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_train.txt
#
# 2) Fix 4 files for 2 classes 
#  mcmr> bash scripts/fix_path_in_index_files.sh /home/ubuntu/research/datasets/Pascal3D "aeroplane  bus"  "test train" 
#        cp /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_test.txt /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_test.txt.backup
#        awk -v p=/home/ubuntu/research/datasets/Pascal3D/aeroplane/annotations/  -F'/' '{print p $NF}' /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_test.txt > /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_test.txt
#        cp /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_train.txt /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_train.txt.backup
#        awk -v p=/home/ubuntu/research/datasets/Pascal3D/aeroplane/annotations/  -F'/' '{print p $NF}' /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_train.txt > /home/ubuntu/research/datasets/Pascal3D/aeroplane/aeroplane_train.txt
#        cp /home/ubuntu/research/datasets/Pascal3D/bus/bus_test.txt /home/ubuntu/research/datasets/Pascal3D/bus/bus_test.txt.backup
#        awk -v p=/home/ubuntu/research/datasets/Pascal3D/bus/annotations/  -F'/' '{print p $NF}' /home/ubuntu/research/datasets/Pascal3D/bus/bus_test.txt > /home/ubuntu/research/datasets/Pascal3D/bus/bus_test.txt
#        cp /home/ubuntu/research/datasets/Pascal3D/bus/bus_train.txt /home/ubuntu/research/datasets/Pascal3D/bus/bus_train.txt.backup
#        awk -v p=/home/ubuntu/research/datasets/Pascal3D/bus/annotations/  -F'/' '{print p $NF}' /home/ubuntu/research/datasets/Pascal3D/bus/bus_train.txt > /home/ubuntu/research/datasets/Pascal3D/bus/bus_train.txt


#for multiple index files 
class_list="$2"
mode_list="$3"
for class in $class_list;
do
for mode in $mode_list;
    do
        in_index_file="$1/${class}/${class}_${mode}.txt"
        backup_index_file="${in_index_file}.backup"
        current_path="$1/${class}/annotations/"
        echo "cp $in_index_file $backup_index_file"
        cp $in_index_file $backup_index_file

        echo  "awk -v p="$current_path"  -F'/' '{print p "'$'"NF}' $in_index_file > $in_index_file"
        awk -v p="$current_path"  -F'/' '{print p $NF}' $backup_index_file  > $in_index_file
    done
done 



