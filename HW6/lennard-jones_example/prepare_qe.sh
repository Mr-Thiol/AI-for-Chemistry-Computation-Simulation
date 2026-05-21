exec/calQ-lasp.x -f cluster_na38_ssw/allstr.arc -cut 10 -peri F
for i in Q2E Q4E Q6E; do
  mv $i "$i"_ssw
done
exec/calQ-lasp.x -f cluster_na38_nvt/allstr.arc -cut 10 -peri F
for i in Q2E Q4E Q6E; do
  mv $i "$i"_nvt
done
