#!/bin/bash
# 专为 LJ55 体系修改的数据清洗与打包脚本 (路径已修复)

\rm -rf allstr.arc allfor.arc

# 1. 剔除相似结构 (Screening)
for i in cluster_lj55_ssw cluster_lj55_nvt; do
  if [ -d "$i" ]; then
    cd $i
      # 修复：改为 ../exec/screen.x 
      ../exec/screen.x -f allstr.arc -tol 0.01
      echo "$i is done"
    cd ..
  else
    echo "警告: 找不到文件夹 $i"
  fi
done

# 2. 合并有效结构
for i in cluster_lj55_ssw cluster_lj55_nvt; do
  if [ -f "$i/screen.arc" ]; then
    cat $i/screen.arc >> allstr.arc
  fi
done

# 3. 计算单点能并转换成 LASP 训练集格式
\rm -rf spe trainlj
mkdir spe
if [ -f "allstr.arc" ]; then
  mv allstr.arc spe/lasp.str
  cp exec/lasp.in.spe spe/lasp.in
  cd spe
  # 修复：改为 ../exec/lasp
  ../exec/lasp
  echo "单点能 (Single point energy) 计算完成"
  
  # 修复：改为 ../exec/arc2train.x
  ../exec/arc2train.x
  mkdir ../trainlj
  mv Train* ../trainlj
  cd ..
  cp exec/lasp.in.train trainlj/lasp.in
  cp exec/adjust_factor trainlj/
  
  cd trainlj
  a=`grep ner TrainStr.txt | wc -l`
  echo "Ntrain $a" >> lasp.in
  echo "训练集已成功准备在 trainlj/ 目录下！"
else
  echo "错误: 没有生成最终的 allstr.arc 文件，请检查前面步骤是否成功。"
fi