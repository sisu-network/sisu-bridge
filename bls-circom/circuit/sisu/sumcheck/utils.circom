pragma circom 2.1.7;

include "../configs.gen.circom";

function calc_sumcheck__transcript_size(sumcheck_index) {
    var n_sumchecks = get_sumcheck__n_sumchecks(sumcheck_index);
    var n_worker_rounds = get_sumcheck__n_worker_rounds(sumcheck_index);
    var n_master_rounds = get_sumcheck__n_master_rounds(sumcheck_index);
    var n_rounds = n_worker_rounds + n_master_rounds;

    if (n_rounds == 0) {
        return 0;
    }

    // For every sumcheck, we has ONE sum_all and n_rounds polynomials (each
    // polynomials has the size of 3).
    return n_sumchecks * (1 + 3*n_rounds);
}

function extract_sumcheck__sum_all_range(sumcheck_index) {
    var range[2];
    range[0] = 0;
    range[1] = get_sumcheck__n_sumchecks(sumcheck_index);

    return range;
}

function extract_sumcheck__quadratic_poly_start_index(sumcheck_index, round, k) {
    var n_sumchecks = get_sumcheck__n_sumchecks(sumcheck_index);
  
    return n_sumchecks + 3*round*n_sumchecks + k*3;
}
