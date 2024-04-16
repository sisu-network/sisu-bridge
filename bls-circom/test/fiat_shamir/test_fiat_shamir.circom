pragma circom 2.1.7;
include "../../circuit/sisu/fiat_shamir_hash.circom";

template Test(n, p) {
    signal input seed;
    signal input input0;
    signal input input1;
    signal input input2[3];
    signal input input3[3];
    signal input input4;

    signal input output0;
    signal input output1[3];
    signal input output2;
    signal input output3[2];
    signal input output4;

    // One to One
    component fiat_shamir_0 = FiatShamirHashOneToOne(n, p);
    fiat_shamir_0.x <== input0;
    fiat_shamir_0.fs_in_r <== 0;
    fiat_shamir_0.fs_in_round <== 0;

    output0 === fiat_shamir_0.out;

    // One to Many
    component fiat_shamir_1 = FiatShamirHashOneToMany(n, p, 3);
    fiat_shamir_1.x <== input1;
    fiat_shamir_1.fs_in_r <== fiat_shamir_0.fs_out_r;
    fiat_shamir_1.fs_in_round <== fiat_shamir_0.fs_out_round;

    output1[0] === fiat_shamir_1.out[0];
    output1[1] === fiat_shamir_1.out[1];
    output1[2] === fiat_shamir_1.out[2];

    // Many To One
    component fiat_shamir_2 = FiatShamirHashManyToOne(n, p, 3);
    fiat_shamir_2.xs[0] <== input2[0];
    fiat_shamir_2.xs[1] <== input2[1];
    fiat_shamir_2.xs[2] <== input2[2];
    fiat_shamir_2.fs_in_r <== fiat_shamir_1.fs_out_r;
    fiat_shamir_2.fs_in_round <== fiat_shamir_1.fs_out_round;

    output2 === fiat_shamir_2.out;

    // Many To Many
    component fiat_shamir_3 = FiatShamirHashManyToMany(n, p, 3, 2);
    fiat_shamir_3.xs[0] <== input3[0];
    fiat_shamir_3.xs[1] <== input3[1];
    fiat_shamir_3.xs[2] <== input3[2];
    fiat_shamir_3.fs_in_r <== fiat_shamir_2.fs_out_r;
    fiat_shamir_3.fs_in_round <== fiat_shamir_2.fs_out_round;

    
    output3[0] === fiat_shamir_3.out[0];
    output3[1] === fiat_shamir_3.out[1];

    // One to One with Seed
    component mimc = MimcMultiple(n, p, 2);
    mimc.x[0] <== seed;
    mimc.x[1] <== input4;

    component fiat_shamir_4 = FiatShamirHashOneToOne(n, p);
    fiat_shamir_4.x <== mimc.out;
    fiat_shamir_4.fs_in_r <== fiat_shamir_3.fs_out_r;
    fiat_shamir_4.fs_in_round <== fiat_shamir_3.fs_out_round;

    output4 === fiat_shamir_4.out;
}

component main {public [input0]} = Test(64, 18446744069414584321);
