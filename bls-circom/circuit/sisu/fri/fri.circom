pragma circom  2.1.7;

include "../merkle_tree.circom";
include "../fr_bn254.circom";
include "./fold.circom";
include "./utils.circom";

template MultiFRIRecoverRandomPoints(vpd_index) {
    signal input fiat_shamir_seed;
    signal input fiat_shamir_in_r;
    signal input fiat_shamir_in_round;
    signal input commitment_transcript[calc_fri__commitment_transcript_size(vpd_index)];

    var n_layers = get_fri__n_layers(vpd_index);
    signal output fiat_shamir_out_r;
    signal output fiat_shamir_out_round;
    signal output out[n_layers];

    component mimc_seed;
    component fiat_shamir_hash[n_layers];
    for (var i = 0; i < n_layers; i++) {
        fiat_shamir_hash[i] = FiatShamirHashOneToOne();
        if (i == 0) {
			mimc_seed = MimcMultiple(2);
			mimc_seed.in[0] <== fiat_shamir_seed;
			mimc_seed.in[1] <== 0;

			fiat_shamir_hash[i].in <== mimc_seed.out;
			fiat_shamir_hash[i].in_r <== fiat_shamir_in_r;
			fiat_shamir_hash[i].in_round <== fiat_shamir_in_round;
        } else {
            fiat_shamir_hash[i].in <== extract_fri__commitment_transcript__root(commitment_transcript, vpd_index, i);
            fiat_shamir_hash[i].in_r <== fiat_shamir_hash[i-1].out_r;
            fiat_shamir_hash[i].in_round <== fiat_shamir_hash[i-1].out_round;
        }

        out[i] <== fiat_shamir_hash[i].out;
    }

    fiat_shamir_out_r <== fiat_shamir_hash[n_layers-1].out_r;
    fiat_shamir_out_round <== fiat_shamir_hash[n_layers-1].out_round;
}

template FoldAllEvaluations(vpd_index, layer_index) {
    var num_workers = get_fri__n_workers(vpd_index);

    signal input z_evaluations[num_workers];
    signal input op_z_evaluations[num_workers];
    signal input r;
    signal input index;

    signal output out[num_workers];

    var generators[get_fri__precomputed_domain_generators_size(vpd_index, layer_index)]
        = get_fri__precomputed_domain_generators(vpd_index, layer_index);
    var generator = generators[0];

    component inverse_domain_element = GetFRIDomainElementInverse(vpd_index, layer_index);
    inverse_domain_element.pow <== index;

    // r * (2^-1 * domain[i]^-1);
    signal inverse_two_domain_element <== inverse_two_frbn254() * inverse_domain_element.out;
    signal r__inverse_two__inverse_domain_element <== r * inverse_two_domain_element;

    component folds[num_workers];
    for (var worker_index = 0; worker_index < num_workers; worker_index++) {
        folds[worker_index]= FRIFoldEvaluation(vpd_index, layer_index);
        folds[worker_index].positive_evaluation <== z_evaluations[worker_index];
        folds[worker_index].negative_evaluation <== op_z_evaluations[worker_index];
        folds[worker_index].r__inverse_two__inverse_domain_element <== r__inverse_two__inverse_domain_element;

        out[worker_index] <== folds[worker_index].out;
    }
}

template VerifyQuery(vpd_index, layer_index) {
    var n_workers = get_fri__n_workers(vpd_index);
    var query_path_size = calc_fri__query_path_size(vpd_index, layer_index);

    signal input transcript[get_fri__query_size(vpd_index, layer_index)];
    signal input root;
    signal input index;

    signal output out[n_workers];

    component verify_multi_merkle_tree = VerifyMultiMerkleTreeMimc(n_workers, query_path_size);
    verify_multi_merkle_tree.root <== root;
    verify_multi_merkle_tree.index <== index;

    var evaluation_range[2] = extract_fri__query_evaluation_range(vpd_index);
    for (var i = evaluation_range[0]; i < evaluation_range[1]; i++) {
        verify_multi_merkle_tree.evaluations[i - evaluation_range[0]] <== transcript[i];
    }

    verify_multi_merkle_tree.evaluations_root <== extract_fri__query_evaluation_hash(transcript, vpd_index);
    var path_range[2] = extract_fri__query_path_range(vpd_index, layer_index);
    for (var i = path_range[0]; i < path_range[1]; i++) {
        verify_multi_merkle_tree.path[i - path_range[0]] <== transcript[i];
    }

    // Extract evaluation from transcript.
    for (var i = evaluation_range[0]; i < evaluation_range[1]; i++) {
        out[i - evaluation_range[0]] <== transcript[i];
    }
}

template VerifyMultiFRI(vpd_index) {
    var commitment_transcript_size = calc_fri__commitment_transcript_size(vpd_index);
    var query_transcript_size = calc_fri__query_transcript_size(vpd_index);
    var n_workers = get_fri__n_workers(vpd_index);
    var n_layers = get_fri__n_layers(vpd_index);

    signal input commitment_transcript[commitment_transcript_size];
    signal input query_transcript[query_transcript_size];
    signal input first_z_evaluations[n_workers];
    signal input first_op_z_evaluations[n_workers];
    signal input trusted_r[n_layers];
    signal input index;

    component sum[n_layers];
    component next_index[n_layers];
    component op_index[n_layers];
    component verify_z_queries[n_layers];
    component verify_op_z_queries[n_layers];
    for (var layer_index = 0; layer_index < n_layers; layer_index ++) {
        if (layer_index == 0) {
            sum[layer_index] = FoldAllEvaluations(vpd_index, layer_index);
            sum[layer_index].r <== trusted_r[layer_index];
            sum[layer_index].index <== index;
            for (var worker_index = 0; worker_index < n_workers; worker_index++) {
                sum[layer_index].z_evaluations[worker_index] <== first_z_evaluations[worker_index];
                sum[layer_index].op_z_evaluations[worker_index]<== first_op_z_evaluations[worker_index];
            }

            next_index[layer_index] = GetFRINextDomainIndex(vpd_index, layer_index);
            next_index[layer_index].in <== index;
        } else {
            var query_path_size = calc_fri__query_path_size(vpd_index, layer_index);

            op_index[layer_index] = GetFRIOpppositeDomainIndex(vpd_index, layer_index);
            op_index[layer_index].in <== next_index[layer_index-1].out;

            var z_query_range[2] = extract_fri__query_transcript__z_query_range(vpd_index, layer_index);
            verify_z_queries[layer_index] = VerifyQuery(vpd_index, layer_index);
            verify_z_queries[layer_index].root <== extract_fri__commitment_transcript__root(commitment_transcript, vpd_index, layer_index);
            verify_z_queries[layer_index].index <== next_index[layer_index-1].out;
            for (var i = z_query_range[0]; i < z_query_range[1]; i++) {
                verify_z_queries[layer_index].transcript[i - z_query_range[0]] <== query_transcript[i]; 
            }

            var op_z_query_range[2] = extract_fri__query_transcript__op_z_query_range(vpd_index, layer_index);
            verify_op_z_queries[layer_index] = VerifyQuery(vpd_index, layer_index);
            verify_op_z_queries[layer_index].root <== extract_fri__commitment_transcript__root(commitment_transcript, vpd_index, layer_index);
            verify_op_z_queries[layer_index].index <== op_index[layer_index].out;
            for (var i = op_z_query_range[0]; i < op_z_query_range[1]; i++) {
                verify_op_z_queries[layer_index].transcript[i - op_z_query_range[0]] <== query_transcript[i]; 
            }

            sum[layer_index] = FoldAllEvaluations(vpd_index, layer_index);
            sum[layer_index].r <== trusted_r[layer_index];
            sum[layer_index].index <== next_index[layer_index-1].out;
            for (var worker_index = 0; worker_index < n_workers; worker_index++) {
                verify_z_queries[layer_index].out[worker_index] === sum[layer_index - 1].out[worker_index];
                sum[layer_index].z_evaluations[worker_index] <== verify_z_queries[layer_index].out[worker_index];
                sum[layer_index].op_z_evaluations[worker_index] <== verify_op_z_queries[layer_index].out[worker_index];
            }

            next_index[layer_index] = GetFRINextDomainIndex(vpd_index, layer_index);
            next_index[layer_index].in <== next_index[layer_index-1].out;
        }
    }

    for (var worker_index = 0; worker_index < n_workers; worker_index++) {
        extract_fri__commitment_transcript__final_constant(commitment_transcript, vpd_index, worker_index)
        === sum[n_layers - 1].out[worker_index];
    }
}
