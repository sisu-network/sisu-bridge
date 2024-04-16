#!/bin/bash

echo "****GENERATING Tau****"
start=`date +%s`
snarkjs powersoftau new bn128 15 pot12_0000.ptau -v
snarkjs powersoftau contribute pot12_0000.ptau pot12.ptau --name="First contribution" -v
snarkjs powersoftau prepare phase2 pot12.ptau pot12_final.ptau -v
end=`date +%s`
echo "DONE ($((end-start))s)"

rm pot12_0000.ptau
mv pot12.ptau ./common
mv pot12_final.ptau ./common
