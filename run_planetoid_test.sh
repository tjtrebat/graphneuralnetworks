#!/bin/bash

for i in {1..10}
do
    python planetoid_test.py --model SpecNet --k $i --dataset PubMed
done