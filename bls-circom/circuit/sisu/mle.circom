pragma circom 2.1.7;


template IdentityMLE_2(num_vars) {
    signal input in1[num_vars];
    signal input in2[num_vars];

    signal output out;

    // result *= (a*b) + (1-a)*(1-b)
    //         = ab + 1 - a - b + ab
    //         = 2ab - a - b + 1
    signal tmp_out[num_vars];
    signal tmp[num_vars];
    for (var i = 0; i < num_vars; i++) {
        tmp[i] <== 2 * in1[i] * in2[i] - in1[i] - in2[i] + 1;
        if (i == 0) {
            tmp_out[i] <== tmp[i];
        } else {
            tmp_out[i] <== tmp_out[i-1] * tmp[i]; 
        }
    }

    out <== tmp_out[num_vars - 1];
}

template IdentityMLE_3(num_vars) {
    signal input in1[num_vars];
    signal input in2[num_vars];
    signal input in3[num_vars];


    signal output out;

    // result *= (a*b*c) + (1-a)*(1-b)*(1-c)
    //         = abc + (1 - a - b + ab) * (1-c)
    //         = abc + 1 - a - b + ab - c + ac + bc - abc
    //         = ab + ac + bc - a - b - c + 1
    //         = a * (b + c - 1) + bc - b - c + 1
    //         = (a - 1) * (b + c - 1) + bc
    signal tmp_out[num_vars];
    signal tmp0[num_vars];
    signal tmp1[num_vars];
    for (var i = 0; i < num_vars; i++) {
        tmp0[i] <== (in1[i] - 1) * (in2[i] + in3[i] - 1);
        tmp1[i] <== in2[i] * in3[i];
        if (i == 0) {
            tmp_out[i] <== tmp0[i] + tmp1[i];
        } else {
            tmp_out[i] <== tmp_out[i-1] * (tmp0[i] + tmp1[i]); 
        }
    }

    out <== tmp_out[num_vars - 1];
}
