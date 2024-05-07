
#Inputs:  copy_report.files <test name> <postifix>
# save  report files (symmetry, cad distances) from the test with given name to  files with given postfix
# *.txt,  *.npy --> *_<postfix>.txt,  *-<postix>.npy
#Examples
# mcmr_root> bash  scripts/copy_report_files.sh small prev
# mcmr_root> bash  scripts/copy_report_files.sh _ prev #full scale tests 

find save/*_$1  -name "cad_mesh_distances.txt"  | while read filename
do
   echo -e  cp $filename "${filename%.*}_$2.txt"
   cp $filename "${filename%.*}_$2.txt"
done 

find save/*_$1  -name "cad_mesh_distances.npy"  | while read filename
do
   echo -e cp $filename "${filename%.*}_$2.npy"
   cp $filename "${filename%.*}_$2.npy"
done 
