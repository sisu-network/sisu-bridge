pragma circom 2.1.7;

include "../mimc.circom";

template SumQuadraticPoly(n) {
    signal input a0[n];
    signal input a1[n];
    signal input a2[n];

    signal output out_a0;
    signal output out_a1;
    signal output out_a2;

    assert(n <= 4);
    if (n == 1) {
        out_a0 <== a0[0];
        out_a1 <== a1[0];
        out_a2 <== a2[0];
    } else if (n == 2) {
        out_a0 <== a0[0] + a0[1];
        out_a1 <== a1[0] + a1[1];
        out_a2 <== a2[0] + a2[1];
    } else if (n == 3) {
        out_a0 <== a0[0] + a0[1] + a0[2];
        out_a1 <== a1[0] + a1[1] + a1[2];
        out_a2 <== a2[0] + a2[1] + a2[2];
    } else if (n == 4) {
        out_a0 <== a0[0] + a0[1] + a0[2] + a0[3];
        out_a1 <== a1[0] + a1[1] + a1[2] + a1[3];
        out_a2 <== a2[0] + a2[1] + a2[2] + a2[3];
    }
}

template QuadraticPolyEvaluate() {
    signal input a0;
    signal input a1;
    signal input a2;
    signal input x;
    signal output out;

    signal x2 <== x*x;
    signal a2x2 <== a2*x2;
    
    out <== a0 + a1 * x + a2x2;
}

template QuadraticPolyAtZero_Add_PolyAtOne() {
    signal input a0;
    signal input a1;
    signal input a2;
    signal output out;

    
    //   f(0) + f(1)
    // = a + b*0 + c*0^2 + a + b*1 + c*1^2
    // = 2*a + b + c
    
    out <== 2 * a0 + a1 + a2;
}

template QuadraticPolyHashCoeffs() {
    signal input a0;
    signal input a1;
    signal input a2;
    signal output out;

    component mimc = MimcMultiple(3);
    mimc.in[0] <== a0;
    mimc.in[1] <== a1;
    mimc.in[2] <== a2;

    out <== mimc.out;
}
