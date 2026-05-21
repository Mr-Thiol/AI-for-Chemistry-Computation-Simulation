#!/bin/bash
# 专为 LJ55 体系修改的 Steinhardt 序参量计算脚本

if [ -f "cluster_lj55_ssw/allstr.arc" ]; then
  exec/calQ-lasp.x -f cluster_lj55_ssw/allstr.arc -cut 10 -peri F
  for i in Q2E Q4E Q6E; do
    if [ -f "$i" ]; then
      mv $i "$i"_ssw
    fi
  done
  echo "SSW 体系的 Q-E 参数计算完毕。"
else
  echo "警告: 找不到 cluster_lj55_ssw/allstr.arc"
fi

if [ -f "cluster_lj55_nvt/allstr.arc" ]; then
  exec/calQ-lasp.x -f cluster_lj55_nvt/allstr.arc -cut 10 -peri F
  for i in Q2E Q4E Q6E; do
    if [ -f "$i" ]; then
      mv $i "$i"_nvt
    fi
  done
  echo "NVT 体系的 Q-E 参数计算完毕。"
else
  echo "警告: 找不到 cluster_lj55_nvt/allstr.arc"
fi