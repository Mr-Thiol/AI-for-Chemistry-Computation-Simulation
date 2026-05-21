\rm -rf allstr.arc allfor.arc
for i in slab_na16 crystal_na12;do
  cd $i
    ../exec/screen.x -f allstr.arc -tol 0.03
    echo "$i" "is done"
  cd ..
done
for i in cluster_na38_ssw cluster_na38_nvt;do
  cd $i
    ../exec/screen.x -f allstr.arc -tol 0.01
    echo "$i" "is done"
  cd ..
done
for i in slab_na16 crystal_na12 cluster_na38_ssw cluster_na38_nvt; do 
  cat $i/screen.arc >>allstr.arc
done
\rm -f spe trainlj
mkdir spe
mv allstr.arc spe/lasp.str
cp exec/lasp.in.spe spe/lasp.in
cd spe
../exec/lasp
echo "Single point energy computation is done"
../exec/arc2train.x
mkdir ../trainlj
mv Train* ../trainlj
cd ..
cp exec/lasp.in.train trainlj/lasp.in
cp exec/adjust_factor trainlj/
cd trainlj
a=`grep ner TrainStr.txt|wc -l`
echo "Ntrain $a" >> lasp.in
