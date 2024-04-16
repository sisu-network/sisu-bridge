pragma circom 2.1.7;

include "../fr_bn254.circom";
include "../fiat_shamir_hash.circom";
include "./quadratic_poly.circom";
include "./utils.circom";


// Verify product sumcheck.
template VerifyMultiProductSumCheck(sumcheck_index) {
	var n_sumchecks = get_sumcheck__n_sumchecks(sumcheck_index);
	var n_worker_rounds = get_sumcheck__n_worker_rounds(sumcheck_index);
	var n_master_rounds = get_sumcheck__n_master_rounds(sumcheck_index);

	signal input transcript[calc_sumcheck__transcript_size(sumcheck_index)];
	signal input fiat_shamir_seed;
	signal input fiat_shamir_in_r;
	signal input fiat_shamir_in_round;

	signal output random_points[n_worker_rounds + n_master_rounds];
	signal output fiat_shamir_out_r;
	signal output fiat_shamir_out_round;
	signal output final_value;
	signal output sum_over_boolean_hypercube;

	var sum_all_range[2] = extract_sumcheck__sum_all_range(sumcheck_index);

	component sum_sum_all = SumSumcheckValue(n_sumchecks);
	signal sum_all[n_sumchecks];
	for (var i = sum_all_range[0]; i < sum_all_range[1]; i++) {
		sum_all[i - sum_all_range[0]] <== transcript[i];
		sum_sum_all.in[i - sum_all_range[0]] <== transcript[i];
	}

	component g_r[n_worker_rounds + n_master_rounds][n_sumchecks];
	component f0_add_f1[n_worker_rounds + n_master_rounds][n_sumchecks];

	component final_poly[n_worker_rounds + n_master_rounds];
	component hash_poly[n_worker_rounds + n_master_rounds];
	component fiat_shamir_hash[n_worker_rounds + n_master_rounds];

	component mimc_seed;

	for (var i = 0; i < n_worker_rounds + n_master_rounds; i++) {
		fiat_shamir_hash[i] = FiatShamirHashOneToOne();
		final_poly[i] = SumQuadraticPoly(n_sumchecks);
		for (var k = 0; k < n_sumchecks; k++) {
			var poly_start_index = extract_sumcheck__quadratic_poly_start_index(sumcheck_index, i, k);
			final_poly[i].a0[k] <== transcript[poly_start_index + 0];
			final_poly[i].a1[k] <== transcript[poly_start_index + 1];
			final_poly[i].a2[k] <== transcript[poly_start_index + 2];

			f0_add_f1[i][k] = QuadraticPolyAtZero_Add_PolyAtOne();
			f0_add_f1[i][k].a0 <== transcript[poly_start_index + 0];
			f0_add_f1[i][k].a1 <== transcript[poly_start_index + 1];
			f0_add_f1[i][k].a2 <== transcript[poly_start_index + 2];

			// We put the quadratic poly evaluation here, but not pass the
			// input x. We will pass the input after generating the random point.
			g_r[i][k] = QuadraticPolyEvaluate();
			g_r[i][k].a0 <== transcript[poly_start_index + 0];
			g_r[i][k].a1 <== transcript[poly_start_index + 1];
			g_r[i][k].a2 <== transcript[poly_start_index + 2];

			if (i == 0) {
				sum_all[k] === f0_add_f1[i][k].out;
			} else {
				g_r[i-1][k].out === f0_add_f1[i][k].out;
			}
		}

		hash_poly[i] = QuadraticPolyHashCoeffs();
		hash_poly[i].a0 <== final_poly[i].out_a0;
		hash_poly[i].a1 <== final_poly[i].out_a1;
		hash_poly[i].a2 <== final_poly[i].out_a2;

		if (i == 0) {
			mimc_seed = MimcMultiple(2);
			mimc_seed.in[0] <== fiat_shamir_seed;
			mimc_seed.in[1] <== hash_poly[i].out;

			fiat_shamir_hash[i].in <== mimc_seed.out;
			fiat_shamir_hash[i].in_r <== fiat_shamir_in_r;
			fiat_shamir_hash[i].in_round <== fiat_shamir_in_round;
		} else {
			fiat_shamir_hash[i].in <== hash_poly[i].out;
			fiat_shamir_hash[i].in_r <== fiat_shamir_hash[i-1].out_r;
			fiat_shamir_hash[i].in_round <== fiat_shamir_hash[i-1].out_round;
		}

		// Pass input to evaluations.
		for (var k = 0; k < n_sumchecks; k++) {
			g_r[i][k].x <== fiat_shamir_hash[i].out;
		}
	}

	component sum_final_value = SumSumcheckValue(n_sumchecks);
	for (var k = 0; k < n_sumchecks; k++) {
		sum_final_value.in[k] <== g_r[n_worker_rounds + n_master_rounds - 1][k].out;
	}

	// Output random points
	for (var i = 0; i < n_worker_rounds + n_master_rounds; i++) {
		random_points[i] <== fiat_shamir_hash[n_worker_rounds + n_master_rounds - i - 1].out;
	}

	// We freezing the fiat_shamir engine after first n_worker_rounds.
	fiat_shamir_out_r <== fiat_shamir_hash[n_worker_rounds-1].out_r;
	fiat_shamir_out_round <== fiat_shamir_hash[n_worker_rounds-1].out_round;

	// Pass the output.
	sum_over_boolean_hypercube <== sum_sum_all.out;
	final_value <== sum_final_value.out;
}

template SumSumcheckValue(n) {
	signal input in[n];
	signal output out;

	assert(n <= 4);
	if (n == 1) {
		out <== in[0];
	} else if (n == 2) {
		out <== in[0] + in[1];
	} else if (n == 3) {
		out <== in[0] + in[1] + in[2];
	} else if (n == 4) {
		out <== in[0] + in[1] + in[2] + in[3];
	}
}
