#!/bin/bash

# View R1CS details: snarkjs r1cs info circuit.r1cs

CIRCUIT_NAME=$1
BUILD_DIR=../../build
CIRCUIT_TEST=test_$CIRCUIT_NAME
GLOBAL_TAU=../../common/pot12_final.ptau
# GLOBAL_TAU=/home/ubuntu/tau/powersOfTau28_hez_final_25.ptau
JS_MEM=2048000
#RELEASE_NODE=/home/ubuntu/code/node/out/Release/node
RELEASE_NODE=node
#SNARKJS_PATH=/home/ubuntu/code/snarkjs/cli.js
SNARKJS_PATH=snarkjs
SNARK_JS="$RELEASE_NODE --trace-gc --trace-gc-ignore-scavenger --max-old-space-size=$JS_MEM --initial-old-space-size=$JS_MEM --no-global-gc-scheduling --no-incremental-marking --max-semi-space-size=1024 --initial-heap-size=$JS_MEM --expose-gc $SNARKJS_PATH"
RAPID_SNARK=/home/ubuntu/code/rapidsnark/package/bin/prover


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
    echo "****GENERATING ZKEY****"
    start=`date +%s`
    $SNARK_JS groth16 setup "$BUILD_DIR"/"$CIRCUIT_TEST".r1cs "$GLOBAL_TAU" "$BUILD_DIR"/"$CIRCUIT_TEST".zkey
    end=`date +%s`
    echo "DONE ($((end-start))s)"

    # We are doing a short-cut here. We do not contribute to this key. Production requires another
    # contribution.
    echo "****VERIFYING FINAL ZKEY****"
    start=`date +%s`
    $SNARK_JS zkey verify -verbose "$BUILD_DIR"/"$CIRCUIT_TEST".r1cs "$GLOBAL_TAU" "$BUILD_DIR"/"$CIRCUIT_TEST".zkey
    end=`date +%s`
    echo "DONE ($((end-start))s)"

    echo "****EXPORTING VKEY****"
    start=`date +%s`
    $SNARK_JS zkey export verificationkey "$BUILD_DIR"/"$CIRCUIT_TEST".zkey "$BUILD_DIR"/vkey.json
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
    ls $ROOT_DIR/input.json
    ls ./"$CIRCUIT_TEST"
    ./"$CIRCUIT_TEST" $ROOT_DIR/input.json witness.wtns
    end=`date +%s`
    echo "DONE ($((end-start))s)"

    # Generate witness into json
    $RELEASE_NODE $SNARKJS_PATH wej witness.wtns witness.json

    echo "****GENERATING PROOF FOR SAMPLE INPUT****"
    start=`date +%s`
    $RAPID_SNARK "$BUILD_DIR"/"$CIRCUIT_TEST".zkey "$BUILD_DIR"/"$CIRCUIT_TEST"_cpp/witness.wtns "$BUILD_DIR"/proof.json $ROOT_DIR/public.json
    # npx snarkjs groth16 prove "$BUILD_DIR"/"$CIRCUIT_TEST".zkey "$BUILD_DIR"/"$CIRCUIT_TEST"_cpp/witness.wtns "$BUILD_DIR"/proof.json $ROOT_DIR/public.json
    end=`date +%s`
    echo "DONE ($((end-start))s)"

    echo "****VERIFYING PROOF FOR SAMPLE INPUT****"
    start=`date +%s`
    $SNARK_JS groth16 verify "$BUILD_DIR"/vkey.json $ROOT_DIR/public.json "$BUILD_DIR"/proof.json
    end=`date +%s`
    echo "DONE ($((end-start))s)"
}

info() {
    $SNARK_JS r1cs info "$BUILD_DIR"/"$CIRCUIT_TEST".r1cs
}

build
gen_key
gen_proof
info
