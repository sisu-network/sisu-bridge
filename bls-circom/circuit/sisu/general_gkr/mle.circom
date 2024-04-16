pragma circom  2.1.7;

include "../configs.gen.circom";
include "../mle.circom";
include "../sparse_mle.circom";

template GeneralGKRSparseMleEvaluateSingleExt(gkr_index, src_layer_index, dst_layer_index, ext_index) {
    var single_g_num_vars = get_general_gkr__single_subcircuit_num_vars(gkr_index, src_layer_index);
    var single_x_num_vars = get_general_gkr__single_subcircuit_num_vars(gkr_index, src_layer_index + 1);
    var single_y_num_vars = get_general_gkr__single_subcircuit_subset_num_vars(gkr_index, src_layer_index, dst_layer_index);

    var point_len = single_g_num_vars + single_x_num_vars + single_y_num_vars;

    var eval_len = get_general_gkr__ext__evaluations_size(gkr_index, src_layer_index, dst_layer_index, ext_index);
    signal input points[point_len];
    signal output out;

    if (eval_len == 0) {
        out <== 0;
    } else {
        var evaluations[eval_len] = get_general_gkr__ext__evaluations(gkr_index, src_layer_index, dst_layer_index, ext_index);

        var prev_output_indexes[get_general_gkr__ext__prev_output_indexes_size(gkr_index, src_layer_index, dst_layer_index, ext_index)]
            = get_general_gkr__ext__prev_output_indexes(gkr_index, src_layer_index, dst_layer_index, ext_index);
        var last_positions[get_general_gkr__ext__last_positions_size(gkr_index, src_layer_index, dst_layer_index, ext_index)]
            = get_general_gkr__ext__last_positions(gkr_index, src_layer_index, dst_layer_index, ext_index);
        var flattened_lens[get_general_gkr__ext__flattened_size_size(gkr_index, src_layer_index, dst_layer_index, ext_index)]
            = get_general_gkr__ext__flattened_size(gkr_index, src_layer_index, dst_layer_index, ext_index);
        var old_idxes[get_general_gkr__ext__old_indexes_size(gkr_index, src_layer_index, dst_layer_index, ext_index)]
            = get_general_gkr__ext__old_indexes(gkr_index, src_layer_index, dst_layer_index, ext_index);

        component mle = SparseMleEvaluate(eval_len, point_len, flattened_lens,
            old_idxes, prev_output_indexes, last_positions);

        for (var i = 0; i < point_len; i++) {
            mle.points[i] <== points[i];
        }
        for (var i = 0; i < eval_len; i++) {
            mle.evaluations[i] <== evaluations[i];
        }

        out <== mle.out;
    }
}

template GeneralGKRSparseMleEvaluateLayer(gkr_index, layer_index) {
    var num_rounds = get_general_gkr__num_rounds(gkr_index);
    var extra_num_vars = calc_general__extra_num_vars(gkr_index);
    var g_num_vars = calc_general_gkr__g_num_vars(gkr_index, layer_index);
    var x_num_vars = calc_general_gkr__x_num_vars(gkr_index, layer_index);
    var max_y_num_vars = calc_general_gkr__max_y_num_vars(gkr_index, layer_index);
    signal input g[g_num_vars];
    signal input u[x_num_vars];
    signal input v[max_y_num_vars];

    signal output const_out[num_rounds - layer_index];
    signal output mul_out[num_rounds - layer_index];
    signal output forward_x_out[num_rounds - layer_index];
    signal output forward_y_out[num_rounds - layer_index];

    component const_mle[num_rounds - layer_index];
    component mul_mle[num_rounds - layer_index];
    component forward_x_mle[num_rounds - layer_index];
    component forward_y_mle[num_rounds - layer_index];
    component identity_extra[num_rounds - layer_index];

    var single_g_num_vars = get_general_gkr__single_subcircuit_num_vars(gkr_index, layer_index);
    var single_x_num_vars = get_general_gkr__single_subcircuit_num_vars(gkr_index, layer_index + 1);
    var g_x_combined_point_size = single_g_num_vars + single_x_num_vars;

    signal g_x_combined_point[g_x_combined_point_size];
    for (var i = 0; i < single_g_num_vars; i++) {
        g_x_combined_point[i] <== g[extra_num_vars + i];
    }
    for (var i = 0; i < single_x_num_vars; i++) {
        g_x_combined_point[single_g_num_vars + i] <== u[extra_num_vars + i];
    }

    for (var dst_layer_index = layer_index + 1; dst_layer_index < num_rounds + 1; dst_layer_index++) {
        var single_y_num_vars = get_general_gkr__single_subcircuit_subset_num_vars(gkr_index, layer_index, dst_layer_index);

        const_mle[dst_layer_index - layer_index - 1] = GeneralGKRSparseMleEvaluateSingleExt(gkr_index, layer_index, dst_layer_index, 0);
        mul_mle[dst_layer_index - layer_index - 1] = GeneralGKRSparseMleEvaluateSingleExt(gkr_index, layer_index, dst_layer_index, 1);
        forward_x_mle[dst_layer_index - layer_index - 1] = GeneralGKRSparseMleEvaluateSingleExt(gkr_index, layer_index, dst_layer_index, 2);
        forward_y_mle[dst_layer_index - layer_index - 1] = GeneralGKRSparseMleEvaluateSingleExt(gkr_index, layer_index, dst_layer_index, 3);

        for (var i = 0; i < single_y_num_vars; i++) {
            const_mle[dst_layer_index - layer_index - 1].points[i] <== v[extra_num_vars + single_y_num_vars - i - 1];
            mul_mle[dst_layer_index - layer_index - 1].points[i] <== v[extra_num_vars + single_y_num_vars - i - 1];
            forward_x_mle[dst_layer_index - layer_index - 1].points[i] <== v[extra_num_vars + single_y_num_vars - i - 1];
            forward_y_mle[dst_layer_index - layer_index - 1].points[i] <== v[extra_num_vars + single_y_num_vars - i - 1];
        }

        for (var i = 0; i < g_x_combined_point_size; i++) {
            const_mle[dst_layer_index - layer_index - 1].points[single_y_num_vars + i]
                <== g_x_combined_point[g_x_combined_point_size - i - 1];

            mul_mle[dst_layer_index - layer_index - 1].points[single_y_num_vars + i]
                <== g_x_combined_point[g_x_combined_point_size - i - 1];

            forward_x_mle[dst_layer_index - layer_index - 1].points[single_y_num_vars + i]
                <== g_x_combined_point[g_x_combined_point_size - i - 1];

            forward_y_mle[dst_layer_index - layer_index - 1].points[single_y_num_vars + i]
                <== g_x_combined_point[g_x_combined_point_size - i - 1];
        }

        identity_extra[dst_layer_index - layer_index - 1] = IdentityMLE_3(extra_num_vars);
        for (var i = 0; i < extra_num_vars; i++) {
            identity_extra[dst_layer_index-layer_index-1].in1[i] <== g[i];
            identity_extra[dst_layer_index-layer_index-1].in2[i] <== u[i];
            identity_extra[dst_layer_index-layer_index-1].in3[i] <== v[i];
        }

        const_out[dst_layer_index - layer_index - 1] <== identity_extra[dst_layer_index - layer_index - 1].out * const_mle[dst_layer_index - layer_index - 1].out;
        mul_out[dst_layer_index - layer_index - 1] <== identity_extra[dst_layer_index - layer_index - 1].out * mul_mle[dst_layer_index - layer_index - 1].out;
        forward_x_out[dst_layer_index - layer_index - 1] <== identity_extra[dst_layer_index - layer_index - 1].out * forward_x_mle[dst_layer_index - layer_index - 1].out;
        forward_y_out[dst_layer_index - layer_index - 1] <== identity_extra[dst_layer_index - layer_index - 1].out * forward_y_mle[dst_layer_index - layer_index - 1].out;
    }
}


template GeneralGKRSparseMleEvaluateReverseExt(gkr_index, src_layer_index, dst_layer_index) {
    var single_subset_num_vars = get_general_gkr__single_subcircuit_subset_num_vars(gkr_index, src_layer_index, dst_layer_index);
    var single_real_num_vars = get_general_gkr__single_subcircuit_num_vars(gkr_index, dst_layer_index);

    var point_len = single_subset_num_vars + single_real_num_vars;

    var eval_len = get_general_gkr__reverse_subset__ext__evaluations_size(gkr_index, src_layer_index, dst_layer_index);
    signal input points[point_len];
    signal output out;

    if (eval_len == 0) {
        out <== 0;
    } else {
        var evaluations[eval_len] = get_general_gkr__reverse_subset__ext__evaluations(gkr_index, src_layer_index, dst_layer_index);

        var prev_output_indexes[get_general_gkr__reverse_subset__ext__prev_output_indexes_size(gkr_index, src_layer_index, dst_layer_index)]
            = get_general_gkr__reverse_subset__ext__prev_output_indexes(gkr_index, src_layer_index, dst_layer_index);
        var last_positions[get_general_gkr__reverse_subset__ext__last_positions_size(gkr_index, src_layer_index, dst_layer_index)]
            = get_general_gkr__reverse_subset__ext__last_positions(gkr_index, src_layer_index, dst_layer_index);
        var flattened_lens[get_general_gkr__reverse_subset__ext__flattened_size_size(gkr_index, src_layer_index, dst_layer_index)]
            = get_general_gkr__reverse_subset__ext__flattened_size(gkr_index, src_layer_index, dst_layer_index);
        var old_idxes[get_general_gkr__reverse_subset__ext__old_indexes_size(gkr_index, src_layer_index, dst_layer_index)]
            = get_general_gkr__reverse_subset__ext__old_indexes(gkr_index, src_layer_index, dst_layer_index);

        component mle = SparseMleEvaluate(eval_len, point_len, flattened_lens,
            old_idxes, prev_output_indexes, last_positions);

        for (var i = 0; i < point_len; i++) {
            mle.points[i] <== points[i];
        }
        for (var i = 0; i < eval_len; i++) {
            mle.evaluations[i] <== evaluations[i];
        }

        out <== mle.out;
    }
}
