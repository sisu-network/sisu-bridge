pragma circom 2.1.7;

include "../configs.gen.circom";
include "../fr_bn254.circom";


template GetVPDDomainElement(vpd_index) {
    signal input pow;
    signal output out;

    var precomputed_generators_size = get_vpd__precomputed_ldt_domain_generators_size(vpd_index);
    var precomputed_generators[precomputed_generators_size] = get_vpd__precomputed_ldt_domain_generators(vpd_index);

    component get_domain_element = GetDomainElement(precomputed_generators_size);
    get_domain_element.pow <== pow;
    for (var i = 0; i < precomputed_generators_size; i++) {
        get_domain_element.precomputed_generators[i] <== precomputed_generators[i];
    }

    out <== get_domain_element.out;
}

template GetVPDDomainElementInverse(vpd_index) {
    signal input pow;
    signal output out;

    var precomputed_generators_size = get_vpd__precomputed_ldt_domain_generators_size(vpd_index);
    var precomputed_generators[precomputed_generators_size] = get_vpd__precomputed_ldt_domain_generators(vpd_index);

    component get_domain_element = GetDomainElementInverse(precomputed_generators_size);
    get_domain_element.pow <== pow;
    for (var i = 0; i < precomputed_generators_size; i++) {
        get_domain_element.precomputed_generators[i] <== precomputed_generators[i];
    }

    out <== get_domain_element.out;
}

template GetVPDDomainElementPowSumcheckDomainSize(vpd_index) {
    signal input pow;
    signal output out;

    var sumcheck_domain_size = get_vpd__sumcheck_domain_size(vpd_index);
    signal pow_mul_sumcheck_size <== pow * sumcheck_domain_size;

    var ldt_domain_size = get_vpd__ldt_domain_size(vpd_index);
    signal reduce_pow <-- pow_mul_sumcheck_size % ldt_domain_size;
    signal tmp <-- pow_mul_sumcheck_size \ ldt_domain_size;
    pow_mul_sumcheck_size === tmp * ldt_domain_size + reduce_pow;

    component get_domain_element = GetVPDDomainElement(vpd_index);
    get_domain_element.pow <== reduce_pow;

    out <== get_domain_element.out;
}

template GetVPDOpppositeDomainIndex(vpd_index) {
    signal input in;
    signal output out;

    var half_domain_size = get_vpd__ldt_domain_size(vpd_index) \ 2;
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
