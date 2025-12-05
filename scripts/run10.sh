#!/bin/bash

for i in {10..24}; do
    echo "Running trial $i..."
    bash run_inference.sh 
done
