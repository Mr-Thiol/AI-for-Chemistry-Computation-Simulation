for i in crystal_na12  slab_na16  cluster_na38_ssw  cluster_na38_nvt  ;do
  cd $i
    mpirun -np 4 ../exec/lasp 
    echo $i " is done"
  cd ..
done
