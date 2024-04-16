pragma circom  2.1.7;

include "./domain.circom";
include "../fr_bn254.circom";

template FRIFoldEvaluation(vpd_index, layer_index) {
    signal input r__inverse_two__inverse_domain_element;
    signal input positive_evaluation;
    signal input negative_evaluation;

    signal output out;

    signal sub_evaluation <== positive_evaluation - negative_evaluation;
    signal sum_evaluation <== positive_evaluation + negative_evaluation;

    // out = r * (postive - negative) * (inverse_two * inverse_domain_e)
    // out += (positive + negative) * inverse_two;
    var inverse_two = inverse_two_frbn254();

    signal out0 <== r__inverse_two__inverse_domain_element * sub_evaluation;
    signal out1 <== sum_evaluation * inverse_two;
    out <== out0 + out1;
}