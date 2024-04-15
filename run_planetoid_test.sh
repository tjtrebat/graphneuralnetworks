#!/bin/bash

for i in {1..5}
do
    python planetoid_test.py --model SpecNet --k $i --dataset Cora
done