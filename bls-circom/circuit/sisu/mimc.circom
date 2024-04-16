pragma circom 2.1.7;

include "./fr_bn254.circom";

// Permuation polynomial: FrBN254.MODULUS % 5 == 2 --> x^5
//
// But we can use a lower power because we do this operation many times
// (128 times).
//
// Currently, x^3 is enough.
template HashOne() {
    signal input in;
    signal output out;

    // Calculate x^3
    signal x2 <== in * in;
    out <== x2 * in;
}

template MimcMultiple(n) {
    signal input in[n];
    signal output out;

    var K[128] = get_mimc_k();
    var D[8]   = get_mimc_d();

    var num_repetitions = 128 \ n;
    if (num_repetitions < 2) {
        num_repetitions = 2;
    }

    component hash_one[num_repetitions][n];
    for (var repetition_index = 0; repetition_index < num_repetitions; repetition_index++) {
        var d = D[repetition_index % 8];
        var start = repetition_index % n;
        for (var i = 0; i < n; i++) {
            var k_index = (repetition_index * n + i) % 128;
            var v_index = (start + d * i) % n;

            hash_one[repetition_index][i] = HashOne();
            if (i == 0) {
                if (repetition_index == 0) {
                    hash_one[repetition_index][i].in <== in[v_index] + K[k_index];
                } else {
                    hash_one[repetition_index][i].in
                        <== hash_one[repetition_index-1][n-1].out + in[v_index] + K[k_index];
                }
            } else {
                hash_one[repetition_index][i].in
                    <== hash_one[repetition_index][i-1].out + in[v_index] + K[k_index];
            }
        }
    }

    out <== hash_one[num_repetitions - 1][n - 1].out;
}
