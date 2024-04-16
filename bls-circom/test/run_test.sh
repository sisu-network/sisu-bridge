#!/bin/bash

# View R1CS details: snarkjs r1cs info circuit.r1cs

CIRCUIT_NAME=$1
BUILD_DIR=../../build
CIRCUIT_TEST=$CIRCUIT_NAME
# CIRCUIT_TEST=test_$CIRCUIT_NAME
# if [ ! -z "$2" ]
#   then
#     CIRCUIT_TEST=$2
# fi
GLOBAL_TAU=../../common/pot12_final.ptau
JS_MEM=8000
NODE_COMMAND=`NODE_ENV=production NODE_OPTIONS="--trace-gc --trace-gc-ignore-scavenger --max-old-space-size=$JS_MEM --initial-old-space-size=$JS_MEM --no-global-gc-scheduling --no-incremental-marking --max-semi-space-size=1024 --initial-heap-size=$JS_MEM --expose-gc"`

if [ ! -d "$BUILD_DIR" ]; then
    echo "No build directory found. Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

ROOT_DIR=$PWD

build() {
    echo "****COMPILING CIRCUIT****"
    start=`date +%s`
    circom "$CIRCUIT_TEST".circom --O1 --c --sym --r1cs --wasm --output "$BUILD_DIR"
    end=`date +%s`
    echo "DONE ($((end-start))s)"
}

gen_key() {
    echo "****GENERATING ZKEY 0****"
    start=`date +%s`
    $NODE_COMMAND npx snarkjs groth16 setup "$BUILD_DIR"/"$CIRCUIT_TEST".r1cs "$GLOBAL_TAU" "$BUILD_DIR"/"$CIRCUIT_TEST"_0.zkey
    end=`date +%s`
    echo "DONE ($((end-start))s)"

    echo "****GENERATING FINAL ZKEY****"
    start=`date +%s`
    $NODE_COMMAND npx snarkjs zkey beacon "$BUILD_DIR"/"$CIRCUIT_TEST"_0.zkey "$BUILD_DIR"/"$CIRCUIT_TEST".zkey 0102030405060708090a0b0c0d0e0f101112231415161718221a1b1c1d1e1f 10 -n="Final Beacon phase2"
    end=`date +%s`
    echo "DONE ($((end-start))s)"

    echo "****VERIFYING FINAL ZKEY****"
    start=`date +%s`
    $NODE_COMMAND npx snarkjs zkey verify -verbose "$BUILD_DIR"/"$CIRCUIT_TEST".r1cs "$GLOBAL_TAU" "$BUILD_DIR"/"$CIRCUIT_TEST".zkey
    end=`date +%s`
    echo "DONE ($((end-start))s)"

    echo "****EXPORTING VKEY****"
    start=`date +%s`
    $NODE_COMMAND npx snarkjs zkey export verificationkey "$BUILD_DIR"/"$CIRCUIT_TEST".zkey "$BUILD_DIR"/vkey.json
    end=`date +%s`
    echo "DONE ($((end-start))s)"
}

gen_proof() {
    # Compile witness generation code
    echo "****COMPILING C++ WITNESS GENERATION CODE****"
    start=`date +%s`
    cd "$BUILD_DIR"/"$CIRCUIT_TEST"_cpp
    make
    end=`date +%s`
    echo "DONE ($((end-start))s)"

    # Generate witness
    echo "****GENERATING WITNESS****"
    start=`date +%s`
    ./"$CIRCUIT_TEST" $ROOT_DIR/input.json witness.wtns
    end=`date +%s`
    echo "DONE ($((end-start))s)"

    # Generate witness into json
    npx snarkjs wej witness.wtns witness.json

    echo "****GENERATING PROOF FOR SAMPLE INPUT****"
    start=`date +%s`
    npx snarkjs groth16 prove "$BUILD_DIR"/"$CIRCUIT_TEST".zkey "$BUILD_DIR"/"$CIRCUIT_TEST"_cpp/witness.wtns "$BUILD_DIR"/proof.json $ROOT_DIR/public.json
    end=`date +%s`
    echo "DONE ($((end-start))s)"

    echo "****VERIFYING PROOF FOR SAMPLE INPUT****"
    start=`date +%s`
    npx snarkjs groth16 verify "$BUILD_DIR"/vkey.json $ROOT_DIR/public.json "$BUILD_DIR"/proof.json
    end=`date +%s`
    echo "DONE ($((end-start))s)"
}

info() {
    snarkjs r1cs info "$BUILD_DIR"/"$CIRCUIT_TEST".r1cs
}

build
gen_key
gen_proof
info
