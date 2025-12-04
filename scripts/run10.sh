#!/bin/bash

for i in {0..9}; do
    echo "Running trial $i..."
    bash run_inference.sh 
done

