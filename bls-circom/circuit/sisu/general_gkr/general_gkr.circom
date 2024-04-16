pragma circom 2.1.7;

include "../sumcheck/sumcheck.circom";
include "../configs.gen.circom";
include "../dense_mle.circom";
include "./utils.circom";
include "./mle.circom";

template Product(num_vars) {
    signal input v[num_vars];

    signal output out;

    if (num_vars == 0) {
        out <== 1;
    } else {
        signal tmp[num_vars];
        for (var i = 0; i < num_vars; i++) {
            if (i == 0) {
                tmp[i] <== v[i];
            } else {
                tmp[i] <== tmp[i-1] * v[i];
            }
        }

        out <== tmp[num_vars - 1];
    }
}

template GeneralGKROracleSum(gkr_index, layer_index) {
    var n_rounds = get_general_gkr__num_rounds(gkr_index);
    var extra_num_vars = calc_general__extra_num_vars(gkr_index);
    var g_num_vars = calc_general_gkr__g_num_vars(gkr_index, layer_index);
    var x_num_vars = calc_general_gkr__x_num_vars(gkr_index, layer_index);
    var max_y_num_vars = calc_general_gkr__max_y_num_vars(gkr_index, layer_index);

    signal input random_g[g_num_vars];
	signal input random_u[x_num_vars];
	signal input random_v[max_y_num_vars];
    signal input w_u;
    signal input w_v[n_rounds - layer_index];

    signal output out;

    component mle = GeneralGKRSparseMleEvaluateLayer(gkr_index, layer_index);
    for (var i = 0; i < g_num_vars; i++) {
        mle.g[i] <== random_g[i];
    }
    for (var i = 0; i < x_num_vars; i++) {
        mle.u[i] <== random_u[i];
    }
    for (var i = 0; i < max_y_num_vars; i++) {
        mle.v[i] <== random_v[i];
    }

    signal w_uv[n_rounds - layer_index];
    signal tmp_mul[n_rounds - layer_index];
    signal tmp_forward_x[n_rounds - layer_index];
    signal tmp_forward_y[n_rounds - layer_index];
    signal oracle_sum[n_rounds - layer_index];
    component remaining_product_v[n_rounds - layer_index];
    for (var i = 0; i < n_rounds - layer_index; i++) {
        var y_num_vars = calc_general_gkr__y_num_vars(gkr_index, layer_index, layer_index + i + 1);

        remaining_product_v[i] = Product(max_y_num_vars - y_num_vars);
        for (var j = y_num_vars; j < max_y_num_vars; j++) {
            remaining_product_v[i].v[j - y_num_vars] <== random_v[j];
        }

        w_uv[i] <== w_u * w_v[i];
        tmp_mul[i] <== mle.mul_out[i] * w_uv[i];
        tmp_forward_x[i] <== mle.forward_x_out[i] * w_u;
        tmp_forward_y[i] <== mle.forward_y_out[i] * w_v[i];
        if (i == 0) {
            oracle_sum[i] <== remaining_product_v[i].out
                            * (mle.const_out[i] + tmp_mul[i] + tmp_forward_x[i] + tmp_forward_y[i]);
        } else {
            oracle_sum[i] <== oracle_sum[i-1] 
                            + remaining_product_v[i].out
                              * (mle.const_out[i] + tmp_mul[i] + tmp_forward_x[i] + tmp_forward_y[i]);
        }
    }

    out <== oracle_sum[n_rounds - layer_index - 1];
}

template VerifyGeneralGKRSumcheck(gkr_index, layer_index) {
    signal input transcript[calc_general_gkr__round_sumcheck_transcript_size(gkr_index, layer_index)];
    signal input w_g;
    signal input fiat_shamir_in_r;
    signal input fiat_shamir_in_round;

    var x_num_vars = calc_general_gkr__x_num_vars(gkr_index, layer_index);
    var max_y_num_vars = calc_general_gkr__max_y_num_vars(gkr_index, layer_index);

    signal output fiat_shamir_out_r;
    signal output fiat_shamir_out_round;
	signal output final_value;
	signal output sum_over_boolean_hypercube;
	signal output random_u[x_num_vars];
	signal output random_v[max_y_num_vars];

    var sumcheck_phase_1_range[2] = extract_general_gkr__round_sumcheck_transcript__phase_1_transcript_size(gkr_index, layer_index);
    component sumcheck_phase_1 = VerifyMultiProductSumCheck(gkr_index * 1000 + layer_index * 10 + 1);
    sumcheck_phase_1.fiat_shamir_seed <== w_g;
    sumcheck_phase_1.fiat_shamir_in_r <== fiat_shamir_in_r;
    sumcheck_phase_1.fiat_shamir_in_round <== fiat_shamir_in_round;
    for (var i = sumcheck_phase_1_range[0]; i < sumcheck_phase_1_range[1]; i++) {
        sumcheck_phase_1.transcript[i - sumcheck_phase_1_range[0]] <== transcript[i];
    }

    var sumcheck_phase_2_range[2] = extract_general_gkr__round_sumcheck_transcript__phase_2_transcript_size(gkr_index, layer_index);
    component sumcheck_phase_2 = VerifyMultiProductSumCheck(gkr_index * 1000 + layer_index * 10 + 2);
    sumcheck_phase_2.fiat_shamir_seed <== w_g;
    sumcheck_phase_2.fiat_shamir_in_r <== sumcheck_phase_1.fiat_shamir_out_r;
    sumcheck_phase_2.fiat_shamir_in_round <== sumcheck_phase_1.fiat_shamir_out_round;
    for (var i = sumcheck_phase_2_range[0]; i < sumcheck_phase_2_range[1]; i++) {
        sumcheck_phase_2.transcript[i - sumcheck_phase_2_range[0]] <== transcript[i];
    }

    fiat_shamir_out_r <== sumcheck_phase_2.fiat_shamir_out_r;
    fiat_shamir_out_round <== sumcheck_phase_2.fiat_shamir_out_round;
    final_value <== sumcheck_phase_2.final_value;
    sum_over_boolean_hypercube <== sumcheck_phase_1.sum_over_boolean_hypercube;

    for (var i = 0; i < x_num_vars; i++) {
        random_u[i] <== sumcheck_phase_1.random_points[i];
    }
    for (var i = 0; i < max_y_num_vars; i++) {
        random_v[i] <== sumcheck_phase_2.random_points[i];
    }
}

template VerifyGeneralGKRRound(gkr_index, layer_index) {
    var n_rounds = get_general_gkr__num_rounds(gkr_index);
    var extra_num_vars = calc_general__extra_num_vars(gkr_index);
    var g_num_vars = calc_general_gkr__g_num_vars(gkr_index, layer_index);
    var x_num_vars = calc_general_gkr__x_num_vars(gkr_index, layer_index);
    var max_y_num_vars = calc_general_gkr__max_y_num_vars(gkr_index, layer_index);
    var max_reverse_subset_num_vars = calc_general_gkr__max_reverse_subset_num_vars(gkr_index, layer_index + 1);

    signal input fiat_shamir_in_r;
    signal input fiat_shamir_in_round;
    signal input transcript[calc_general_gkr__round_transcript_size(gkr_index, layer_index)];
    signal input w_g;
    signal input random_g[g_num_vars];
    signal input alpha[layer_index + 2];
    // Actually, at layer i, we have i+2 claims, but the last two claims just appear in this round.
    signal input prev_claims[layer_index];
    signal input prev_claims_random_points[layer_index][max_reverse_subset_num_vars];

    signal output fiat_shamir_out_r;
    signal output fiat_shamir_out_round;
    signal output next_w_g;
    signal output next_random_g[x_num_vars];
	signal output random_u[x_num_vars];
	signal output random_v[max_y_num_vars];
    signal output w_u;
    signal output w_v[n_rounds - layer_index];

    // RUN PHASE SUMCHECK
    var phase_transcript_range[2] = extract_general_gkr__round_transcript__sumcheck_range(gkr_index, layer_index);
    component verify_gkr_sumcheck = VerifyGeneralGKRSumcheck(gkr_index, layer_index);
    verify_gkr_sumcheck.w_g <== w_g;
    verify_gkr_sumcheck.fiat_shamir_in_r <== fiat_shamir_in_r;
    verify_gkr_sumcheck.fiat_shamir_in_round <== fiat_shamir_in_round;
    for (var i = phase_transcript_range[0]; i < phase_transcript_range[1]; i++) {
        verify_gkr_sumcheck.transcript[i - phase_transcript_range[0]] <== transcript[i];
    }

    // CHECK W_G === SUM_OVER_BOOLEAN_HYPERCUBE
    w_g === verify_gkr_sumcheck.sum_over_boolean_hypercube;

    // EXPORT RANDOM_U, RANDOM_V.
    for (var i = 0; i < x_num_vars; i++) {
        random_u[i] <== verify_gkr_sumcheck.random_u[i];
    }
    for (var i = 0; i < max_y_num_vars; i++) {
        random_v[i] <== verify_gkr_sumcheck.random_v[i];
    }

    // EXPORT W_U, W_V.
    w_u <== extract_general_gkr__round_transcript__w_u(transcript, gkr_index, layer_index);
    var w_v_range[2] = extract_general_gkr__round_transcript__w_v_range(gkr_index, layer_index);
    for (var i = w_v_range[0]; i < w_v_range[1]; i++) {
        w_v[i - w_v_range[0]] <== transcript[i];
    }

    // CHECK ORACLE === SUM_OVER_BOOLEAN_HYPERCUBE
    component oracle_sum = GeneralGKROracleSum(gkr_index, layer_index);
    oracle_sum.w_u <== w_u;
    for (var i = 0; i < n_rounds - layer_index; i++) {
        oracle_sum.w_v[i] <== w_v[i];
    }
    for (var i = 0; i < g_num_vars; i++) {
        oracle_sum.random_g[i] <== random_g[i];
    }
    for (var i = 0; i < x_num_vars; i++) {
        oracle_sum.random_u[i] <== random_u[i];
    }
    for (var i = 0; i < max_y_num_vars; i++) {
        oracle_sum.random_v[i] <== random_v[i];
    }

    oracle_sum.out === verify_gkr_sumcheck.final_value;

    // RUN COMBINING SUMCHECK
    var combining_sumcheck_range[2] = extract_general_gkr__round_transcript__combining_transcript_size(gkr_index, layer_index);
    component combining_sumcheck = VerifyMultiProductSumCheck(gkr_index * 1000 + layer_index * 10 + 3);
    combining_sumcheck.fiat_shamir_seed <== w_g;
    combining_sumcheck.fiat_shamir_in_r <== verify_gkr_sumcheck.fiat_shamir_out_r;
    combining_sumcheck.fiat_shamir_in_round <== verify_gkr_sumcheck.fiat_shamir_out_round;
    for (var i = combining_sumcheck_range[0]; i < combining_sumcheck_range[1]; i++) {
        combining_sumcheck.transcript[i - combining_sumcheck_range[0]] <== transcript[i];
    }
    
    // EXPORT NEXT_RANDOM_G = combining_sumcheck.random_points (next_g_num_vars === x_num_vars).
    //
    // The following value will be calculated at the end of this template
    // NEXT_W_G = FINAL_VALUE / g_r.
    for (var i = 0; i < x_num_vars; i++) {
        next_random_g[i] <== combining_sumcheck.random_points[i];
    }

    // CHECK SUM OVER BOOLEAN HYPERCUBE OF COMBINING SUMCHECK
    signal expected_sum_combining_sumcheck[layer_index+2];
    expected_sum_combining_sumcheck[0] <== alpha[layer_index] * w_v[0];
    expected_sum_combining_sumcheck[1] <== expected_sum_combining_sumcheck[0] + alpha[layer_index + 1] * w_u;
    for (var i = 0; i < layer_index; i++) {
        expected_sum_combining_sumcheck[i + 2] <== expected_sum_combining_sumcheck[i+1] + alpha[i] * prev_claims[i];
    }

    expected_sum_combining_sumcheck[layer_index+1] === combining_sumcheck.sum_over_boolean_hypercube;

    // CHECK FINAL VALUE OF COMBINING SUMCHECK
    signal expected_final_value_combining_sumcheck[layer_index + 2];
    signal g_r[layer_index + 2];
    signal g_r_tmp[layer_index + 1];
    component identity_mle[layer_index + 2];
    component reverse_subset_mle[layer_index + 1];

    // Firstly, handle current w_u.
    identity_mle[layer_index + 1] = IdentityMLE_2(x_num_vars);
    for (var i = 0; i < x_num_vars; i++) {
        identity_mle[layer_index + 1].in1[i] <== random_u[i];
        identity_mle[layer_index + 1].in2[i] <== next_random_g[i];
    }
    g_r[0] <== alpha[layer_index + 1] * identity_mle[layer_index + 1].out;

    // Then, handle all w_v.
    for (var i = 0; i < layer_index + 1; i++) {
        identity_mle[i] = IdentityMLE_2(extra_num_vars);
        for (var j = 0; j < extra_num_vars; j++) {
            if (i == layer_index) { // handle for the random_v of this round.
                identity_mle[i].in1[j] <== random_v[j];
            } else {
                identity_mle[i].in1[j] <== prev_claims_random_points[i][j];
            }

            identity_mle[i].in2[j] <== next_random_g[j];
        }

        reverse_subset_mle[i] = GeneralGKRSparseMleEvaluateReverseExt(gkr_index, i, layer_index+1);
        var single_real_num_vars = get_general_gkr__single_subcircuit_num_vars(gkr_index, layer_index + 1);
        for (var j = 0; j < single_real_num_vars; j++) {
            reverse_subset_mle[i].points[j] <== next_random_g[extra_num_vars + single_real_num_vars - j - 1];
        }

        var single_subset_num_vars = get_general_gkr__single_subcircuit_subset_num_vars(gkr_index, i, layer_index + 1);
        for (var j = 0; j < single_subset_num_vars; j++) {
            if (i == layer_index) { // handle for the random_v of this round.
                reverse_subset_mle[i].points[single_real_num_vars + j]
                    <== random_v[extra_num_vars + single_subset_num_vars - j - 1];
            } else {
                reverse_subset_mle[i].points[single_real_num_vars + j]
                    <== prev_claims_random_points[i][extra_num_vars + single_subset_num_vars - j - 1];
            }
        }

        g_r_tmp[i] <== alpha[i] * identity_mle[i].out;
        g_r[i+1] <== g_r[i] +  g_r_tmp[i] * reverse_subset_mle[i].out;
    }

    // EXPORT NEXT_W_G 
    next_w_g <-- combining_sumcheck.final_value / g_r[layer_index + 1];
    next_w_g * g_r[layer_index + 1] === combining_sumcheck.final_value;

    // EXPORT FIAT_SHAMIR
    fiat_shamir_out_r <== combining_sumcheck.fiat_shamir_out_r;
    fiat_shamir_out_round <== combining_sumcheck.fiat_shamir_out_round;
}

template VerifyGeneralGKRTranscript(gkr_index) { 
    var num_non_zero_outputs = get_general_gkr__num_non_zero_outputs(gkr_index);
	var single_output_size = round_to_next_two_pow(get_general_gkr__num_outputs(gkr_index));
    var num_replicas = get_general_gkr__num_replicas(gkr_index);

	var output_num_vars = calc_general_gkr__g_num_vars(gkr_index, 0);
    var transcript_size = calc_general_gkr__transcript_size(gkr_index);

	signal input fiat_shamir_seed;
	signal input fiat_shamir_in_r;
	signal input fiat_shamir_in_round;
	signal input transcript[transcript_size];

	signal output circuit_output[num_replicas][num_non_zero_outputs];
    signal output last_g;
    signal output last_w_g;

    for (var replica_index = 0; replica_index < num_replicas; replica_index++) {
        var output_range[2] = extract_general_gkr__transcript__output_range(gkr_index, replica_index);
        for (var i = output_range[0]; i < output_range[1]; i++) {
            circuit_output[replica_index][i - output_range[0]] <== transcript[i];
        }
    }

	component mimc_seed_output = MimcMultiple(2);
	mimc_seed_output.in[0] <== fiat_shamir_seed;
	mimc_seed_output.in[1] <== 0;

	component g = FiatShamirHashOneToMany(output_num_vars);
	g.in <== mimc_seed_output.out;
	g.in_r <== fiat_shamir_in_r;
	g.in_round <== fiat_shamir_in_round;

    component output_ext = DenseMajorZeroEvaluate(num_non_zero_outputs, single_output_size, num_replicas);
    for (var i = 0; i < num_replicas; i++) {
        for (var j = 0;  j < num_non_zero_outputs; j++) {
            output_ext.groups[i][j] <== circuit_output[i][j];
        }
    }
    for (var i = 0; i < output_num_vars; i++) {
        output_ext.points[i] <== g.out[i];
    }

	var num_gkr_rounds = get_general_gkr__num_rounds(gkr_index);
	component alpha_hash[num_gkr_rounds];
	component verify_round[num_gkr_rounds];
	for (var round_index = 0; round_index < num_gkr_rounds; round_index++) {
		// alpha
		alpha_hash[round_index] = FiatShamirHashOneToMany(round_index + 2);
        if (round_index == 0) {
		    alpha_hash[round_index].in <== output_ext.out;
			alpha_hash[round_index].in_r <== g.out_r;
			alpha_hash[round_index].in_round <== g.out_round;
        } else {
            alpha_hash[round_index].in <== verify_round[round_index - 1].next_w_g;
			alpha_hash[round_index].in_r <== verify_round[round_index - 1].fiat_shamir_out_r;
			alpha_hash[round_index].in_round <== verify_round[round_index - 1].fiat_shamir_out_round;
        }

		// verify_round
		verify_round[round_index] = VerifyGeneralGKRRound(gkr_index, round_index);
        verify_round[round_index].fiat_shamir_in_r <== alpha_hash[round_index].out_r;
        verify_round[round_index].fiat_shamir_in_round <== alpha_hash[round_index].out_round;

        var g_num_vars = calc_general_gkr__g_num_vars(gkr_index, round_index);
        if (round_index == 0) {
            verify_round[round_index].w_g <== output_ext.out;
            for (var i = 0; i < g_num_vars; i++) {
                verify_round[round_index].random_g[i] <== g.out[i];
            }
        } else {
            verify_round[round_index].w_g <== verify_round[round_index-1].next_w_g;
            for (var i = 0; i < g_num_vars; i++) {
                verify_round[round_index].random_g[i] <== verify_round[round_index-1].next_random_g[i];
            }
        }

        var max_reverse_subset_num_vars = calc_general_gkr__max_reverse_subset_num_vars(gkr_index, round_index+1);
        for (var prev_round_index = 0; prev_round_index < round_index; prev_round_index++) {
            verify_round[round_index].prev_claims[prev_round_index]
                <== verify_round[prev_round_index].w_v[round_index - prev_round_index];

            var y_num_vars = calc_general_gkr__y_num_vars(gkr_index, prev_round_index, round_index+1);
            for (var k = 0; k < y_num_vars; k++) {
                verify_round[round_index].prev_claims_random_points[prev_round_index][k]
                    <== verify_round[prev_round_index].random_v[k];
            }
            for (var k = y_num_vars; k < max_reverse_subset_num_vars; k++) {
                verify_round[round_index].prev_claims_random_points[prev_round_index][k] <== 0;
            }
        }

        for (var j = 0; j < round_index+2; j++) {
            verify_round[round_index].alpha[j] <== alpha_hash[round_index].out[j];
        }

		// transcript
		var range[2] = extract_general_gkr__transcript__round_range(gkr_index, round_index);
		for (var j = range[0]; j < range[1]; j++) {
			verify_round[round_index].transcript[j - range[0]] <== transcript[j];
		}
	}
}
