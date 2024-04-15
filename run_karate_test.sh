#!/bin/bash

for i in {1..10}
do
    python karate_test.py --model SpecNet --k $i
done