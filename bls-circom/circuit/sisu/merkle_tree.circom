pragma circom  2.1.7;

include "../../circomlib/circuits/bitify.circom";
include "./mimc.circom";

template VerifyMerklePathMimc(path_size) {
    signal input root;
    signal input value;
    signal input path[path_size];
    signal input index;

    component index_bit = Num2Bits(path_size);
    index_bit.in <== index;

    component mimc_0[path_size];
    component mimc_1[path_size];
    signal hv0[path_size];
    signal hv1[path_size];
    signal hv[path_size];
    for (var i = 0; i < path_size; i++) {
        mimc_0[i] = MimcMultiple(2);
        mimc_1[i] = MimcMultiple(2);
        if (i == 0) {
            mimc_0[i].in[0] <== value;
            mimc_1[i].in[1] <== value;
        } else {
            mimc_0[i].in[0] <== hv[i-1];
            mimc_1[i].in[1] <== hv[i-1];
        }

        mimc_0[i].in[1] <== path[i];
        mimc_1[i].in[0] <== path[i];
        
        hv0[i] <== (1 - index_bit.out[i]) * mimc_0[i].out;
        hv1[i] <== index_bit.out[i] * mimc_1[i].out;
        hv[i] <== hv0[i] + hv1[i];
    }

    root === hv[path_size - 1];
}

template VerifyMultiMerkleTreeMimc(num_trees, path_size) {
    signal input root;
    signal input path[path_size];
    signal input index;
    signal input evaluations_root;
    signal input evaluations[num_trees];

    signal output out;

    component evaluations_hash = MimcMultiple(num_trees);
    for (var i = 0; i < num_trees; i++) {
        evaluations_hash.in[i] <== evaluations[i];
    }

    evaluations_root === evaluations_hash.out;

    component verify_path = VerifyMerklePathMimc(path_size);
    verify_path.root <== root;
    verify_path.index <== index;
    verify_path.value <== evaluations_root;
    for (var i = 0; i < path_size; i++) {
        verify_path.path[i] <== path[i];
    }
}
