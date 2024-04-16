pragma circom  2.1.7;

include "../../../circomlib/circuits/comparators.circom";
include "../configs.gen.circom";

template GetFRIDomainElement(vpd_index, layer_index) {
    signal input pow;
    signal output out;

    var precomputed_generators_size = get_fri__precomputed_domain_generators_size(vpd_index, layer_index);
    var precomputed_generators[precomputed_generators_size] = get_fri__precomputed_domain_generators(vpd_index, layer_index);

    component get_domain_element = GetDomainElement(precomputed_generators_size);
    get_domain_element.pow <== pow;
    for (var i = 0; i < precomputed_generators_size; i++) {
        get_domain_element.precomputed_generators[i] <== precomputed_generators[i];
    }

    out <== get_domain_element.out;
}

template GetFRIDomainElementInverse(vpd_index, layer_index) {
    signal input pow;
    signal output out;

    var precomputed_generators_size = get_fri__precomputed_domain_generators_size(vpd_index, layer_index);
    var precomputed_generators[precomputed_generators_size] = get_fri__precomputed_domain_generators(vpd_index, layer_index);

    component get_domain_element = GetDomainElementInverse(precomputed_generators_size);
    get_domain_element.pow <== pow;
    for (var i = 0; i < precomputed_generators_size; i++) {
        get_domain_element.precomputed_generators[i] <== precomputed_generators[i];
    }

    out <== get_domain_element.out;
}

template GetFRIOpppositeDomainIndex(vpd_index, layer_index) {
    signal input in;
    signal output out;

    var half_domain_size = get_fri__domain_size(vpd_index, layer_index) \ 2;
    var domain_log_size = log2(half_domain_size) + 1;

    component lt = LessThan(domain_log_size);
    lt.in[0] <== in;
    lt.in[1] <== half_domain_size;

    signal tmp0 <== in + half_domain_size;
    signal tmp1 <== in - half_domain_size;

    signal out0 <== lt.out * tmp0;
    signal out1 <== (1-lt.out) * tmp1;

    out <== out0 + out1;
}

template GetFRINextDomainIndex(vpd_index, layer_index) {
    signal input in;
    signal output out;

    var domain_size = get_fri__domain_size(vpd_index, layer_index);
    var domain_log_size = log2(domain_size);

    component lt = LessThan(domain_log_size);
    lt.in[0] <== in * 2;
    lt.in[1] <== domain_size;

    signal tmp0 <== in * 2;
    signal tmp1 <== in * 2 - domain_size;

    signal out0 <== lt.out * tmp0;
    signal out1 <== (1-lt.out) * tmp1;

    out <== (out0 + out1) / 2;
}