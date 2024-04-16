pragma circom 2.1.7;

include "../../../circomlib/circuits/bitify.circom";
include "../fiat_shamir_hash.circom";
include "../merkle_tree.circom";
include "../mle.circom";
include "../num_func.circom";
include "../gkr/gkr.circom";
include "../fri/fri.circom";
include "./domain.circom";
include "./utils.circom";


template VPDComputeRationalConstraint(vpd_index) {
    signal input x;
    signal input x_inv;
    signal input x_pow_sumcheck_size;
    signal input q;
    signal input l;
    signal input h;
    signal input worker_output;

    signal output out;

    var domain_size = get_vpd__sumcheck_domain_size(vpd_index);
    var domain_size_inv = get_vpd__sumcheck_domain_size_inv(vpd_index);

    // out = (domain_size * l * q - output - domain_size * z * h) * (domain_size_inv * x_inv);
    //     = x_inv * ((l*q - z*h) - output * domain_size_inv); 
    signal lq <== l * q;

    // z = x_pow_sumcheck_size - 1
    // z*h = x_pow_sumcheck_size * h - h
    signal zh <== x_pow_sumcheck_size * h - h;
    signal tmp <== lq - zh - worker_output * domain_size_inv;
    out <== x_inv * tmp;
}


template VerifyVPDFirstSingleQuery(vpd_index) {
    signal input index;
    signal input root;
    signal input query[calc_vpd__first_query__single_query_size(vpd_index)];

    signal output evaluations[get_vpd__num_workers(vpd_index)];

    var num_trees = get_vpd__num_workers(vpd_index);
    var path_size = get_vpd__first_query_path_size(vpd_index);
    component multi_merkle_tree = VerifyMultiMerkleTreeMimc(num_trees, path_size);
    multi_merkle_tree.root <== root;
    multi_merkle_tree.index <== index;

    var evaluations_range[2] = extract_vpd__first_single_query__evaluations_range(vpd_index);
    for (var i = evaluations_range[0]; i < evaluations_range[1]; i++) {
        multi_merkle_tree.evaluations[i - evaluations_range[0]] <== query[i];
        evaluations[i - evaluations_range[0]] <== query[i];
    }

    multi_merkle_tree.evaluations_root <== extract_vpd__first_single_query__evaluations_hash(query, vpd_index);

    var path_range[2] = extract_vpd__first_single_query__path_range(vpd_index);
    for (var i = path_range[0]; i < path_range[1]; i++) {
        multi_merkle_tree.path[i - path_range[0]] <== query[i];
    }
}


template VerifyVPDFirstQuery(vpd_index) {
    signal input positive_index;
    signal input negative_index;
    signal input root;
    signal input query[calc_vpd__first_query_size(vpd_index)];

    signal output positive_evaluations[get_vpd__num_workers(vpd_index)];
    signal output negative_evaluations[get_vpd__num_workers(vpd_index)];

    component verify_positive = VerifyVPDFirstSingleQuery(vpd_index);
    verify_positive.index <== positive_index;
    verify_positive.root <== root;

    var positive_range[2] = extract_vpd__first_query__single_query_range(vpd_index, 0);
    for (var i = positive_range[0]; i < positive_range[1]; i++) {
        verify_positive.query[i - positive_range[0]] <== query[i];
    }

    component verify_negative = VerifyVPDFirstSingleQuery(vpd_index);
    verify_negative.index <== negative_index;
    verify_negative.root <== root;

    var negative_range[2] = extract_vpd__first_query__single_query_range(vpd_index, 1);
    for (var i = negative_range[0]; i < negative_range[1]; i++) {
        verify_negative.query[i - negative_range[0]] <== query[i];
    }

    for (var i = 0; i < get_vpd__num_workers(vpd_index); i++) {
        positive_evaluations[i] <== verify_positive.evaluations[i];
        negative_evaluations[i] <== verify_negative.evaluations[i];
    }
}

template VerifySingleRepetition(vpd_index) {
    var first_query_size = calc_vpd__first_query_size(vpd_index);

    signal input fri_commitment_transcript[calc_fri__commitment_transcript_size(vpd_index)];
    signal input fri_query_transcript[calc_fri__query_transcript_size(vpd_index)];
    signal input fri_random_points[get_fri__n_layers(vpd_index)];

    signal input l_root;
    signal input h_root;
    signal input l_first_query[first_query_size];
    signal input h_first_query[first_query_size];
    signal input positive_q;
    signal input negative_q;
    signal input positive_index;
    signal input negative_index;
    signal input worker_outputs[get_vpd__num_workers(vpd_index)];

    component positive_x = GetVPDDomainElement(vpd_index);
    positive_x.pow <== positive_index;

    component positive_x_inv = GetVPDDomainElementInverse(vpd_index);
    positive_x_inv.pow <== positive_index;

    component positive_x_pow_sumcheck_size = GetVPDDomainElementPowSumcheckDomainSize(vpd_index);
    positive_x_pow_sumcheck_size.pow <== positive_index;

    component negative_x = GetVPDDomainElement(vpd_index);
    negative_x.pow <== negative_index;

    component negative_x_inv = GetVPDDomainElementInverse(vpd_index);
    negative_x_inv.pow <== negative_index;

    component negative_x_pow_sumcheck_size = GetVPDDomainElementPowSumcheckDomainSize(vpd_index);
    negative_x_pow_sumcheck_size.pow <== negative_index;

    component verify_l = VerifyVPDFirstQuery(vpd_index);
    verify_l.root <== l_root;
    verify_l.positive_index <== positive_index;
    verify_l.negative_index <== negative_index;

    component verify_h = VerifyVPDFirstQuery(vpd_index);
    verify_h.root <== h_root;
    verify_h.positive_index <== positive_index;
    verify_h.negative_index <== negative_index;

    for (var i = 0; i < calc_vpd__first_query_size(vpd_index); i++) {
        verify_l.query[i] <== l_first_query[i];
        verify_h.query[i] <== h_first_query[i];
    }

    var n_workers = get_vpd__num_workers(vpd_index);
    component compute_positive_p[n_workers];
    component compute_negative_p[n_workers];
    for (var i = 0; i < n_workers; i++) {
        compute_positive_p[i] = VPDComputeRationalConstraint(vpd_index);
        compute_positive_p[i].x <== positive_x.out;
        compute_positive_p[i].x_inv <== positive_x_inv.out;
        compute_positive_p[i].x_pow_sumcheck_size <== positive_x_pow_sumcheck_size.out;
        compute_positive_p[i].q <== positive_q;
        compute_positive_p[i].l <== verify_l.positive_evaluations[i];
        compute_positive_p[i].h <== verify_h.positive_evaluations[i];
        compute_positive_p[i].worker_output <== worker_outputs[i];

        compute_negative_p[i] = VPDComputeRationalConstraint(vpd_index);
        compute_negative_p[i].x <== negative_x.out;
        compute_negative_p[i].x_inv <== negative_x_inv.out;
        compute_negative_p[i].x_pow_sumcheck_size <== negative_x_pow_sumcheck_size.out;
        compute_negative_p[i].q <== negative_q;
        compute_negative_p[i].l <== verify_l.negative_evaluations[i];
        compute_negative_p[i].h <== verify_h.negative_evaluations[i];
        compute_negative_p[i].worker_output <== worker_outputs[i];
    }

    component verify_fri = VerifyMultiFRI(vpd_index);
    for (var i = 0; i < calc_fri__commitment_transcript_size(vpd_index); i++) {
        verify_fri.commitment_transcript[i] <== fri_commitment_transcript[i];
    }

    for (var i = 0; i < calc_fri__query_transcript_size(vpd_index); i++) {
        verify_fri.query_transcript[i] <== fri_query_transcript[i];
    }

    for (var i = 0; i < get_vpd__num_workers(vpd_index); i++) {
        verify_fri.first_z_evaluations[i] <== compute_positive_p[i].out;
        verify_fri.first_op_z_evaluations[i] <== compute_negative_p[i].out;
    }

    for (var i = 0; i < get_fri__n_layers(vpd_index); i++) {
        verify_fri.trusted_r[i] <== fri_random_points[i];
    }

    verify_fri.index <== positive_index;
}


template VerifyMultiVPD(vpd_index) {
    var vpd_input_size = calc_vpd__input_size(vpd_index);
    var worker_input_size = log2(get_vpd__num_workers(vpd_index));

    signal input fiat_shamir_seed;
    signal input fiat_shamir_in_r;
    signal input fiat_shamir_in_round;
    signal input commitment_transcript[calc_vpd__commitment_size()];
    signal input transcript[calc_vpd__transcript_size(vpd_index)];
    signal input vpd_input[vpd_input_size];

    signal output fiat_shamir_out_r;
    signal output fiat_shamir_out_round;
    signal output vpd_output;

    signal l_root <== extract_vpd__commitment__l_root(commitment_transcript);

    var worker_outputs_range[2] = extract_vpd__transcript__output_range(vpd_index);
    signal worker_outputs[get_vpd__num_workers(vpd_index)];
    for (var i = worker_outputs_range[0]; i < worker_outputs_range[1]; i++) {
        worker_outputs[i - worker_outputs_range[0]] <== transcript[i];
    }

    var p_fri_commitment_range[2] = extract_vpd__transcript__fri_commitment_range(vpd_index);
    signal p_fri_commitment[calc_fri__commitment_transcript_size(vpd_index)];
    for (var i = p_fri_commitment_range[0]; i < p_fri_commitment_range[1]; i++) {
        p_fri_commitment[i - p_fri_commitment_range[0]] <== transcript[i];
    }

    signal h_root <== extract_vpd__transcript__h_root(transcript, vpd_index);

    var fri_n_layers = get_fri__n_layers(vpd_index);
    component fri_random_points = MultiFRIRecoverRandomPoints(vpd_index);
    fri_random_points.fiat_shamir_seed <== fiat_shamir_seed;
    fri_random_points.fiat_shamir_in_r <== fiat_shamir_in_r;
    fri_random_points.fiat_shamir_in_round <== fiat_shamir_in_round;
    for (var i = 0; i < calc_fri__commitment_transcript_size(vpd_index); i++) {
        fri_random_points.commitment_transcript[i] <== p_fri_commitment[i];
    }

    var n_repetitions = get_vpd__num_repetitions(vpd_index);
    component fiat_shamir_input = MimcMultiple(2);
    fiat_shamir_input.in[0] <== extract_fri__commitment_transcript__root(p_fri_commitment, vpd_index, fri_n_layers-1);
    fiat_shamir_input.in[1] <== 0;

    // GENERATE ALL QUERY INDEXES
    component fiat_shamir_indexes[n_repetitions];
    component indexes[n_repetitions];
    component op_indexes[n_repetitions];
    for (var i = 0; i < n_repetitions; i++) {
        fiat_shamir_indexes[i] = FiatShamirHashOneToOne();
        fiat_shamir_indexes[i].in <== fiat_shamir_input.out;
        if (i == 0) {
            fiat_shamir_indexes[i].in_r <== fri_random_points.fiat_shamir_out_r;
            fiat_shamir_indexes[i].in_round <== fri_random_points.fiat_shamir_out_round;
        } else {
            fiat_shamir_indexes[i].in_r <== fiat_shamir_indexes[i-1].out_r;
            fiat_shamir_indexes[i].in_round <== fiat_shamir_indexes[i-1].out_round;
        }

        indexes[i] = HashFieldToInteger(get_vpd__ldt_domain_size(vpd_index));
        indexes[i].in <== fiat_shamir_indexes[i].out;

        op_indexes[i] = GetVPDOpppositeDomainIndex(vpd_index);
        op_indexes[i].in <== indexes[i].out;
    }

    // At this time, we done all FIAT SHAMIR of this protocol. The fiat-shamir
    // of the below FFT GKR is freezing from outside protocol. So we return
    // fiat_shamir output here.
    fiat_shamir_out_r <== fiat_shamir_indexes[n_repetitions-1].out_r;
    fiat_shamir_out_round <== fiat_shamir_indexes[n_repetitions-1].out_round;

    component verify_gkr = VerifyGKRTranscript(calc_fft_gkr_index(vpd_index));

    // SETUP GKR INPUT
    var single_input_size = get_vpd__single_input_size(vpd_index);
    var half_circuit_input_size = round_to_next_two_pow(2 * single_input_size);
    for (var i = 0; i < half_circuit_input_size; i++) { // First half of GKR input is ONE.
        verify_gkr.circuit_input[i] <== 1;
    }
    for (var i = 0; i < single_input_size; i++) { // Second part is [1-t, t]
        verify_gkr.circuit_input[half_circuit_input_size + i*2] <== 1 - vpd_input[worker_input_size + i];
        verify_gkr.circuit_input[half_circuit_input_size + i*2 + 1] <== vpd_input[worker_input_size + i];
    }
    for (var i = half_circuit_input_size + 2*single_input_size; i < 2*half_circuit_input_size; i++) {
        // Padding.
        verify_gkr.circuit_input[half_circuit_input_size + i] <== 0;
    }

    // SETUP GKR TRANSCRIPT
    var q_gkr_transcript_range[2] = extract_vpd__transcript__q_gkr_transcript_range(vpd_index);
    for (var i = q_gkr_transcript_range[0]; i < q_gkr_transcript_range[1]; i++) {
        verify_gkr.transcript[i - q_gkr_transcript_range[0]] <== transcript[i];
    }

    // SETUP GKR VPD RANDOM ACCESS INDEXES.
    for (var i = 0; i < n_repetitions; i++) {
        verify_gkr.vpd_random_access_indexes[i*2] <== indexes[i].out;
        verify_gkr.vpd_random_access_indexes[i*2+1] <== op_indexes[i].out;
    }

    // SETUP GKR FIAT SHAMIR
    verify_gkr.fiat_shamir_seed <== extract_fri__commitment_transcript__root(p_fri_commitment, vpd_index, fri_n_layers-1);
    verify_gkr.fiat_shamir_in_r <== fiat_shamir_indexes[n_repetitions-1].out_r;
    verify_gkr.fiat_shamir_in_round <== fiat_shamir_indexes[n_repetitions-1].out_round;

    // Verify Repetitions.
    var n_workers = get_vpd__num_workers(vpd_index);
    component verify_repetitions[n_repetitions];
    for (var i = 0; i < n_repetitions; i++) {
        verify_repetitions[i] = VerifySingleRepetition(vpd_index);

        for (var j = 0; j < calc_fri__commitment_transcript_size(vpd_index); j++) {
            verify_repetitions[i].fri_commitment_transcript[j] <== p_fri_commitment[j];
        }

        var l_first_query_range[2] = extract_vpd__transcript__l_first_query_range(vpd_index, i);
        for (var j = l_first_query_range[0]; j < l_first_query_range[1]; j++) {
            verify_repetitions[i].l_first_query[j - l_first_query_range[0]] <== transcript[j];
        }

        var h_first_query_range[2] = extract_vpd__transcript__h_first_query_range(vpd_index, i);
        for (var j = h_first_query_range[0]; j < h_first_query_range[1]; j++) {
            verify_repetitions[i].h_first_query[j - h_first_query_range[0]] <== transcript[j];
        }

        var query_range[2] = extract_vpd__transcript__fri_transcript_range(vpd_index, i);
        for (var j = query_range[0]; j < query_range[1]; j++) {
            verify_repetitions[i].fri_query_transcript[j - query_range[0]] <== transcript[j];
        }

        for (var j = 0; j < n_workers; j++) {
            verify_repetitions[i].worker_outputs[j] <== worker_outputs[j];
        }

        
        for (var j = 0; j < get_fri__n_layers(vpd_index); j++) {
            verify_repetitions[i].fri_random_points[j] <== fri_random_points.out[j];
        }

        verify_repetitions[i].l_root <== l_root;
        verify_repetitions[i].h_root <== h_root;
        verify_repetitions[i].positive_q <== verify_gkr.circuit_output[i*2];
        verify_repetitions[i].negative_q <== verify_gkr.circuit_output[i*2+1];
        verify_repetitions[i].positive_index <== indexes[i].out;
        verify_repetitions[i].negative_index <== op_indexes[i].out;
    }

    signal output_tmp[n_workers];
    component num2bits[n_workers];
    component beta[n_workers];
    
    for (var i = 0; i < n_workers; i++) {
        num2bits[i] = Num2Bits(worker_input_size);
        num2bits[i].in <== i;

        beta[i] = IdentityMLE_2(worker_input_size);
        for (var j = 0; j < worker_input_size; j++) {
            beta[i].in1[j] <== num2bits[i].out[j];
            beta[i].in2[j] <== vpd_input[j];
        }

        if (i == 0) {
            output_tmp[i] <== beta[i].out * worker_outputs[i];
        } else {
            output_tmp[i] <== output_tmp[i-1] + beta[i].out * worker_outputs[i];
        }
    }

    vpd_output <== output_tmp[n_workers-1];
}
