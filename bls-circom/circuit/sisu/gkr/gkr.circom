pragma circom 2.1.7;

include "../dense_mle.circom";
include "../fiat_shamir_hash.circom";
include "../num_func.circom";
include "../sumcheck/sumcheck.circom";
include "../configs.gen.circom";
include "./mle.circom";
include "./utils.circom";


template GKROracleSum() {
	signal input w_u;
	signal input w_v;
	signal input constant;
	signal input mul;
	signal input forward_x;
	signal input forward_y;
	signal output out;

	// (constant + mul * w_u * w_v + forward_x * w_u + forward_y * w_v)
	signal w_uv <== w_u * w_v;
	signal mul_w_uv <== mul * w_uv;
	signal forward_x_w_u <== forward_x * w_u;
	signal forward_y_w_v <== forward_y * w_v;
	out <== constant + mul_w_uv + forward_x_w_u + forward_y_w_v;
}

template GKROracleSumAlphaBeta() {
	signal input w_u;
	signal input w_v;
	signal input alpha;
	signal input constant_alpha;
	signal input mul_alpha;
	signal input forward_x_alpha;
	signal input forward_y_alpha;
	signal input beta;
	signal input constant_beta;
	signal input mul_beta;
	signal input forward_x_beta;
	signal input forward_y_beta;

	signal output out;

	component alpha_oracle_sum = GKROracleSum();
	alpha_oracle_sum.w_u <== w_u;
	alpha_oracle_sum.w_v <== w_v;
	alpha_oracle_sum.constant <== constant_alpha;
	alpha_oracle_sum.mul <== mul_alpha;
	alpha_oracle_sum.forward_x <== forward_x_alpha;
	alpha_oracle_sum.forward_y <== forward_y_alpha;

	component beta_oracle_sum = GKROracleSum();
	beta_oracle_sum.w_u <== w_u;
	beta_oracle_sum.w_v <== w_v;
	beta_oracle_sum.constant <== constant_beta;
	beta_oracle_sum.mul <== mul_beta;
	beta_oracle_sum.forward_x <== forward_x_beta;
	beta_oracle_sum.forward_y <== forward_y_beta;

	// alpha * (constant_alpha + mul_alpha * w_u * w_v + forward_x_alpha * w_u + forward_y_alpha * w_v)
	signal alpha_value <== alpha * alpha_oracle_sum.out;
	signal beta_value <== beta * beta_oracle_sum.out;

	// oracle
	out <== alpha_value + beta_value;
}

template VerifyGKRSumcheck(gkr_index, layer_index) {
	var next_num_vars = get_gkr__round_current_num_vars(gkr_index, layer_index+1);

	signal input fiat_shamir_seed;
	signal input fiat_shamir_in_r;
	signal input fiat_shamir_in_round;
	signal input transcript[calc_gkr__round__sumcheck_transcript_size(gkr_index, layer_index)];

	signal output fiat_shamir_out_r;
	signal output fiat_shamir_out_round;
	signal output final_value;
	signal output sum_over_boolean_hypercube;
	signal output random_u[next_num_vars];
	signal output random_v[next_num_vars];

	var phase_1_range[2] = extract_gkr__round_sumcheck__phase_range(gkr_index, layer_index, 1);
	var phase_2_range[2] = extract_gkr__round_sumcheck__phase_range(gkr_index, layer_index, 2);

	component sumcheck_phase_1 = VerifyMultiProductSumCheck(calc_gkr_sumcheck_index(gkr_index, layer_index));
	sumcheck_phase_1.fiat_shamir_seed <== fiat_shamir_seed;
	sumcheck_phase_1.fiat_shamir_in_r <== fiat_shamir_in_r;
	sumcheck_phase_1.fiat_shamir_in_round <== fiat_shamir_in_round;
	for (var i = phase_1_range[0]; i < phase_1_range[1]; i++) {
		sumcheck_phase_1.transcript[i - phase_1_range[0]] <== transcript[i];
	}

	component sumcheck_phase_2 = VerifyMultiProductSumCheck(calc_gkr_sumcheck_index(gkr_index, layer_index));
	sumcheck_phase_2.fiat_shamir_seed <== fiat_shamir_seed;
	sumcheck_phase_2.fiat_shamir_in_r <== sumcheck_phase_1.fiat_shamir_out_r;
	sumcheck_phase_2.fiat_shamir_in_round <== sumcheck_phase_1.fiat_shamir_out_round;
	for (var i = phase_2_range[0]; i < phase_2_range[1]; i++) {
		sumcheck_phase_2.transcript[i - phase_2_range[0]] <== transcript[i];
	}

	fiat_shamir_out_r <== sumcheck_phase_2.fiat_shamir_out_r;
	fiat_shamir_out_round <== sumcheck_phase_2.fiat_shamir_out_round;
	sum_over_boolean_hypercube <== sumcheck_phase_1.sum_over_boolean_hypercube;
	final_value <== sumcheck_phase_2.final_value;

	for (var i = 0; i < next_num_vars; i++) {
		random_u[i] <== sumcheck_phase_1.random_points[i];
		random_v[i] <== sumcheck_phase_2.random_points[i];
	}
}

template VerifyGKRFirstRound(gkr_index) {
	var output_size = get_gkr__output_size(gkr_index);
	var output_num_vars = log2(output_size); 
	
	// next_num_var of layer 0 == current_num_vars of layer 1
	var next_num_vars = get_gkr__round_current_num_vars(gkr_index, 1);

	var vpd_random_len = 0;
	if (gkr_index >= 1000) { // All FFT GKR has the form of 1000 + vpd_index.
		var vpd_index = gkr_index - 1000;
		vpd_random_len = get_vpd__num_repetitions(vpd_index) * 2;
		
		// log("Please open the above commented line and comment the following line");
		// assert(1!=1);
	}

	signal input vpd_random_access_indexes[vpd_random_len];
	signal input fiat_shamir_in_round;
	signal input fiat_shamir_in_r;
	signal input circuit_output[output_size];
	signal input transcript[calc_gkr__round_transcript_size(gkr_index, 0)];
	signal input g[output_num_vars];

	signal output random_u[next_num_vars];
	signal output random_v[next_num_vars];
	signal output w_u;
	signal output w_v;
	signal output fiat_shamir_out_r;
	signal output fiat_shamir_out_round;

	// let output_ext = SisuDenseMultilinearExtension::from_slice(&output);
	// let w_g = output_ext.evaluate(vec![g]);
	component output_ext = DenseEvaluate(output_size);
	for (var i = 0; i < output_size; i++) {
		output_ext.evals[i] <== circuit_output[i];
	}
	for (var i = 0; i < output_num_vars; i++) {
		output_ext.points[i] <== g[i];
	}
	signal w_g <== output_ext.out;

	component verify_sumcheck = VerifyGKRSumcheck(gkr_index, 0);
	verify_sumcheck.fiat_shamir_seed <== w_g;
	verify_sumcheck.fiat_shamir_in_r <== fiat_shamir_in_r;
	verify_sumcheck.fiat_shamir_in_round <== fiat_shamir_in_round;
	var range[2] = extract_gkr__round__sumchecks_range(gkr_index, 0);
	for (var i = range[0]; i < range[1]; i++) {
		verify_sumcheck.transcript[i - range[0]] <== transcript[i];
	}
	for (var i = 0; i < next_num_vars; i++) {
		random_u[i] <== verify_sumcheck.random_u[i];
		random_v[i] <== verify_sumcheck.random_v[i];
	}

	w_g === verify_sumcheck.sum_over_boolean_hypercube;

	// // construct points array.
	// var point_len = output_num_vars + 2 * next_num_vars;
	// signal points[point_len];
	// for (var i = 0; i < output_num_vars; i++) {
	// 	points[i] <== g[i];
	// }
	// for (var i = 0; i < next_num_vars; i++) {
	// 	points[output_num_vars + i] <== verify_sumcheck.random_u[i];
	// }
	// for (var i = 0; i < next_num_vars; i++) {
	// 	points[output_num_vars + next_num_vars + i] <== verify_sumcheck.random_v[i];
	// }


	component layer_mle;
	if (gkr_index < 1000) { // Normal GKR
		layer_mle = GKRSparseMleEvaluateLayer(gkr_index, 0);
		for (var i = 0; i < output_num_vars; i++) {
			layer_mle.g[i] <== g[i];
		}
		for (var i = 0; i < next_num_vars; i++) {
			layer_mle.u[i] <== verify_sumcheck.random_u[i];
			layer_mle.v[i] <== verify_sumcheck.random_v[i];
		}
	} else { // FFT GKR
		// TODO: In case of FFT GKR, we must use a special MLE for the first
		// layer.
		// Use vpd_random_access_indexes for that MLE.
		//
		// Currently, we are using a temporary solution here.
		layer_mle = GKRSparseMleEvaluateLayer(gkr_index, 0);
		for (var i = 0; i < output_num_vars; i++) {
			layer_mle.g[i] <== g[i];
		}
		for (var i = 0; i < next_num_vars; i++) {
			layer_mle.u[i] <== verify_sumcheck.random_u[i];
			layer_mle.v[i] <== verify_sumcheck.random_v[i];
		}
	}

	component oracle_sum = GKROracleSum();
	oracle_sum.w_u <== extract_gkr__round__w_u(transcript, gkr_index, 0);
	oracle_sum.w_v <== extract_gkr__round__w_v(transcript, gkr_index, 0);
	oracle_sum.constant <== layer_mle.const_out;
	oracle_sum.mul <== layer_mle.mul_out;
	oracle_sum.forward_x <== layer_mle.forward_x_out;
	oracle_sum.forward_y <== layer_mle.forward_y_out;

	// oracle_access === final_value
	oracle_sum.out === verify_sumcheck.final_value;

	// Assign output
	w_u <== extract_gkr__round__w_u(transcript, gkr_index, 0);
	w_v <== extract_gkr__round__w_v(transcript, gkr_index, 0);
	fiat_shamir_out_r <== verify_sumcheck.fiat_shamir_out_r;
	fiat_shamir_out_round <== verify_sumcheck.fiat_shamir_out_round;
}

template VerifyGKRRound(gkr_index, layer_index) {
	var current_num_vars = get_gkr__round_current_num_vars(gkr_index, layer_index);
	var next_num_vars = get_gkr__round_next_num_vars(gkr_index, layer_index);

	signal input fiat_shamir_in_r;
	signal input fiat_shamir_in_round;
	signal input prev_w_u;
	signal input prev_w_v;
	signal input transcript[calc_gkr__round_transcript_size(gkr_index, layer_index)];
	signal input prev_u[current_num_vars];
	signal input prev_v[current_num_vars];
	signal input alpha;
	signal input beta;

	signal output random_u[next_num_vars];
	signal output random_v[next_num_vars];
	signal output w_u;
	signal output w_v;
	signal output fiat_shamir_out_r;
	signal output fiat_shamir_out_round;

	// let data = alpha * prev_w_u + beta * prev_w_v;
	signal alpha_w_u <== alpha * prev_w_u;
	signal beta_w_v <== beta * prev_w_v;
	signal alpha_wu__add__beta_wv <== alpha_w_u + beta_w_v;

	// var ts_len = ts_gkr_sumcheck_len(num_vars, linear_sumcheck_count);
	var round_transcript_sumcheck = calc_gkr__round__sumcheck_transcript_size(gkr_index, layer_index);

	component verify_sumcheck = VerifyGKRSumcheck(gkr_index, layer_index);
	verify_sumcheck.fiat_shamir_seed <== alpha_wu__add__beta_wv;
	verify_sumcheck.fiat_shamir_in_r <== fiat_shamir_in_r;
	verify_sumcheck.fiat_shamir_in_round <== fiat_shamir_in_round;

	var range[2] = extract_gkr__round__sumchecks_range(gkr_index, layer_index);
	for (var i = range[0]; i < range[1]; i++) {
		verify_sumcheck.transcript[i - range[0]] <== transcript[i];
	}

	// In our case, fiat_shamir_data == alpha * prev_w_u + beta * prev_w_v.
	// alpha * prev_w_u + beta * prev_w_v == sum_on_boolean_hybercube
	alpha_wu__add__beta_wv === verify_sumcheck.sum_over_boolean_hypercube;

	// Create random point which a concatenation of prev_u (or prev_v), random_u, random_v.
	var point_len = current_num_vars + 2 * next_num_vars;

	// constant_alpha, mul_alpha, forward_x_alpha, forward_y_alpha
	component alpha_mle = GKRSparseMleEvaluateLayer(gkr_index, layer_index);
	for (var i = 0; i < current_num_vars; i++) {
		alpha_mle.g[i] <== prev_u[i];
	}
	for (var i = 0; i < next_num_vars; i++) {
		alpha_mle.u[i] <== verify_sumcheck.random_u[i];
		alpha_mle.v[i] <== verify_sumcheck.random_v[i];
	}

	// constant_beta, mul_beta, forward_x_beta, forward_y_beta
	component beta_mle = GKRSparseMleEvaluateLayer(gkr_index, layer_index);
	for (var i = 0; i < current_num_vars; i++) {
		beta_mle.g[i] <== prev_v[i];
	}
	for (var i = 0; i < next_num_vars; i++) {
		beta_mle.u[i] <== verify_sumcheck.random_u[i];
		beta_mle.v[i] <== verify_sumcheck.random_v[i];
	}

	// oracle_sum = alpha * (constant_alpha + mul_alpha * w_u * w_v + forward_x_alpha * w_u + forward_y_alpha * w_v)
	//             + beta * (constant_beta + mul_beta * w_u * w_v + forward_x_beta * w_u + forward_y_beta * w_v);
	component oracle_sum = GKROracleSumAlphaBeta();
	oracle_sum.w_u <== extract_gkr__round__w_u(transcript, gkr_index, layer_index);
	oracle_sum.w_v <== extract_gkr__round__w_v(transcript, gkr_index, layer_index);

	oracle_sum.alpha <== alpha;
	oracle_sum.constant_alpha <== alpha_mle.const_out;
	oracle_sum.mul_alpha <== alpha_mle.mul_out;
	oracle_sum.forward_x_alpha <== alpha_mle.forward_x_out;
	oracle_sum.forward_y_alpha <== alpha_mle.forward_y_out;

	oracle_sum.beta <== beta;
	oracle_sum.constant_beta <== beta_mle.const_out;
	oracle_sum.mul_beta <== beta_mle.mul_out;
	oracle_sum.forward_x_beta <== beta_mle.forward_x_out;
	oracle_sum.forward_y_beta <== beta_mle.forward_y_out;

	oracle_sum.out === verify_sumcheck.final_value;

	w_u <== oracle_sum.w_u;
	w_v <== oracle_sum.w_v;
	fiat_shamir_out_r <== verify_sumcheck.fiat_shamir_out_r;
	fiat_shamir_out_round <== verify_sumcheck.fiat_shamir_out_round;

	// Assign output.
	for (var i = 0; i < next_num_vars; i++) {
		random_u[i] <== verify_sumcheck.random_u[i];
		random_v[i] <== verify_sumcheck.random_v[i];
	}
}


// If this is FFT GKR (the GKR used in VPD), we should set vpd_index, otherwise,
// set it by -1.
// FFT GKR use a special MLE function in the first round.
template VerifyGKRTranscript(gkr_index) { 
	var input_size = get_gkr__input_size(gkr_index);
	var output_size = get_gkr__output_size(gkr_index);
	var output_num_vars = log2(output_size);

	var vpd_random_len = 0;
	if (gkr_index >= 1000) { // All FFT GKR has the form of 1000 + vpd_index.
		var vpd_index = gkr_index - 1000;
		vpd_random_len = get_vpd__num_repetitions(vpd_index) * 2;

		// log("Please open the above commented line and comment the following line");
		// assert(1!=1);
	}

	signal input vpd_random_access_indexes[vpd_random_len];
	signal input fiat_shamir_seed;
	signal input fiat_shamir_in_r;
	signal input fiat_shamir_in_round;
	signal input circuit_input[input_size];
	signal input transcript[calc_gkr__transcript_size(gkr_index)];

	signal output circuit_output[output_size];

	var output_range[2] = extract_gkr__output_range(gkr_index);
	for (var i = output_range[0]; i < output_range[1]; i++) {
		circuit_output[i - output_range[0]] <== transcript[i];
	}

	component mimc_output = MimcMultiple(output_size);
	for (var i = 0; i < output_size; i++) {
		mimc_output.in[i] <== circuit_output[i];
	}

	component mimc_seed_output = MimcMultiple(2);
	mimc_seed_output.in[0] <== fiat_shamir_seed;
	mimc_seed_output.in[1] <==  mimc_output.out;

	component g = FiatShamirHashOneToMany(output_num_vars);
	g.in <== mimc_seed_output.out;
	g.in_r <== fiat_shamir_in_r;
	g.in_round <== fiat_shamir_in_round;

	// Calculate the first round len
	var first_round_transcript_size = calc_gkr__round_transcript_size(gkr_index, 0);
	component first_round = VerifyGKRFirstRound(gkr_index);
	first_round.fiat_shamir_in_r <== g.out_r;
	first_round.fiat_shamir_in_round <== g.out_round;

	for (var i = 0; i < vpd_random_len; i++) {
		first_round.vpd_random_access_indexes[i] <== vpd_random_access_indexes[i];
	}

	// g
	for (var i = 0; i < output_num_vars; i++) {
		first_round.g[i] <== g.out[i];
	}
	// output
	for (var i = 0; i < output_size; i++) {
		first_round.circuit_output[i] <== circuit_output[i];
	}
	// transcript
	var transcript_range[2] = extract_gkr__round_transcript_range(gkr_index, 0);
	for (var i = transcript_range[0]; i < transcript_range[1]; i++) {
		first_round.transcript[i - transcript_range[0]] <== transcript[i];
	}

	var num_gkr_rounds = get_gkr__n_rounds(gkr_index);
	signal data[num_gkr_rounds - 1];
	component alpha_hash[num_gkr_rounds - 1];
	component beta_hash[num_gkr_rounds - 1];
	component verify_round[num_gkr_rounds - 1];
	for (var i = 0; i < num_gkr_rounds - 1; i++) { // first round was handled before.
		if (i == 0) {
			data[i] <== first_round.w_u + first_round.w_v;
		} else {
			data[i] <== verify_round[i - 1].w_u + verify_round[i - 1].w_v;
		}

		// alpha
		alpha_hash[i] = FiatShamirHashOneToOne();
		alpha_hash[i].in <== data[i];
		if (i == 0) {
			alpha_hash[i].in_r <== first_round.fiat_shamir_out_r;
			alpha_hash[i].in_round <== first_round.fiat_shamir_out_round;
		} else {
			alpha_hash[i].in_r <== verify_round[i - 1].fiat_shamir_out_r;
			alpha_hash[i].in_round <== verify_round[i - 1].fiat_shamir_out_round;
		}

		// beta
		beta_hash[i] = FiatShamirHashOneToOne();
		beta_hash[i].in <== data[i];
		beta_hash[i].in_r <== alpha_hash[i].out_r;
		beta_hash[i].in_round <== alpha_hash[i].out_round;

		// verify_round
		verify_round[i] = VerifyGKRRound(gkr_index, i + 1);
		var current_num_vars = get_gkr__round_current_num_vars(gkr_index, i + 1);

		// in_round, in_r
		verify_round[i].fiat_shamir_in_r <== beta_hash[i].out_r;
		verify_round[i].fiat_shamir_in_round <== beta_hash[i].out_round;

		// prev_w_u, prev_w_v, prev_u[num_vars], prev_v[num_vars]
		if (i == 0) {
			verify_round[i].prev_w_u <== first_round.w_u;
			verify_round[i].prev_w_v <== first_round.w_v;
			for (var j = 0; j < current_num_vars; j++) {
				verify_round[i].prev_u[j] <== first_round.random_u[j];
				verify_round[i].prev_v[j] <== first_round.random_v[j];
			}
		} else {
			verify_round[i].prev_w_u <== verify_round[i - 1].w_u;
			verify_round[i].prev_w_v <== verify_round[i - 1].w_v;
			for (var j = 0; j < current_num_vars; j++) {
				verify_round[i].prev_u[j] <== verify_round[i - 1].random_u[j];
				verify_round[i].prev_v[j] <== verify_round[i - 1].random_v[j];
			}
		}
		// alpha, beta
		verify_round[i].alpha <== alpha_hash[i].out;
		verify_round[i].beta <== beta_hash[i].out;

		// transcript
		var range[2] = extract_gkr__round_transcript_range(gkr_index, i + 1);
		for (var j = range[0]; j < range[1]; j++) {
			verify_round[i].transcript[j - range[0]] <== transcript[j];
		}
	}

	// Verify last W_U, W_V
	component oracle_access_w_u = DenseEvaluate(input_size);
	component oracle_access_w_v = DenseEvaluate(input_size);
	for (var i = 0; i < input_size; i++) {
		oracle_access_w_u.evals[i] <== circuit_input[i];
		oracle_access_w_v.evals[i] <== circuit_input[i];
	}
	for (var i = 0; i < log2(input_size); i++) {
		oracle_access_w_u.points[i] <== verify_round[num_gkr_rounds - 2].random_u[i];
		oracle_access_w_v.points[i] <== verify_round[num_gkr_rounds - 2].random_v[i];
	}
	oracle_access_w_u.out === verify_round[num_gkr_rounds - 2].w_u;
	oracle_access_w_v.out === verify_round[num_gkr_rounds - 2].w_v;
}
