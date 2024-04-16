pragma circom  2.1.7;

include "../configs.gen.circom";
include "../sparse_mle.circom";
include "./mle.gen.circom";

template GKRSparseMleEvaluateSingleExt(gkr_index, layer_index, ext_index) {
    var x_num_vars = 0;
    if (layer_index == 0) {
        var output_size = get_gkr__output_size(gkr_index);
        x_num_vars = log2(output_size);    
    } else {
        x_num_vars = get_gkr__round_current_num_vars(gkr_index, layer_index);
    }
    var next_x_num_vars = get_gkr__round_current_num_vars(gkr_index, layer_index+1);

    var point_len = x_num_vars + 2 * next_x_num_vars;
    signal input points[point_len];
    signal output out;

    if (get_custom_mle__point_size(gkr_index, layer_index, ext_index) > 0) {
        component custom_mle = CustomMLEEvaluate(gkr_index, layer_index, ext_index);
        for (var i = 0; i < point_len; i++) {
            custom_mle.points[i] <== points[i];
        }
        out <== custom_mle.out;
    } else {
        var eval_len = get_gkr__ext__evaluations_size(gkr_index, layer_index, ext_index);
        if (eval_len == 0) {
            out <== 0;
        } else {
            var evaluations[eval_len] = get_gkr__ext__evaluations(gkr_index, layer_index, ext_index);

            var prev_output_indexes[get_gkr__ext__prev_output_indexes_size(gkr_index, layer_index, ext_index)]
                = get_gkr__ext__prev_output_indexes(gkr_index, layer_index, ext_index);
            var last_positions[get_gkr__ext__last_positions_size(gkr_index, layer_index, ext_index)]
                = get_gkr__ext__last_positions(gkr_index, layer_index, ext_index);
            var flattened_lens[get_gkr__ext__flattened_size_size(gkr_index, layer_index, ext_index)]
                = get_gkr__ext__flattened_size(gkr_index, layer_index, ext_index);
            var old_idxes[get_gkr__ext__old_indexes_size(gkr_index, layer_index, ext_index)]
                = get_gkr__ext__old_indexes(gkr_index, layer_index, ext_index);

            component mle = SparseMleEvaluate(eval_len, point_len, flattened_lens,
                old_idxes, prev_output_indexes, last_positions);

            for (var i = 0; i < point_len; i++) {
                mle.points[i] <== points[point_len - i - 1];
            }
            for (var i = 0; i < eval_len; i++) {
                mle.evaluations[i] <== evaluations[i];
            }

            out <== mle.out;
        }
    }
}

template GKRSparseMleEvaluateLayer(gkr_index, layer_index) {
    var x_num_vars = 0;
    if (layer_index == 0) {
        var output_size = get_gkr__output_size(gkr_index);
        x_num_vars = log2(output_size);    
    } else {
        x_num_vars = get_gkr__round_current_num_vars(gkr_index, layer_index);
    }
    var next_x_num_vars = get_gkr__round_current_num_vars(gkr_index, layer_index+1);

    signal input g[x_num_vars];
    signal input u[next_x_num_vars];
    signal input v[next_x_num_vars];

    signal output const_out;
    signal output mul_out;
    signal output forward_x_out;
    signal output forward_y_out;

    var combined_points_size = x_num_vars + 2 * next_x_num_vars;
    signal combined_points[combined_points_size];
    for (var i = 0; i < x_num_vars; i++) {
        combined_points[i] <== g[i];
    }

    for (var i = 0; i < next_x_num_vars; i++) {
        combined_points[x_num_vars + i] <== u[i];
        combined_points[x_num_vars + next_x_num_vars + i] <== v[i];
    }

    component const_mle = GKRSparseMleEvaluateSingleExt(gkr_index, layer_index, 0);
    component mul_mle = GKRSparseMleEvaluateSingleExt(gkr_index, layer_index, 1);
    component forward_x_mle = GKRSparseMleEvaluateSingleExt(gkr_index, layer_index, 2);
    component forward_y_mle = GKRSparseMleEvaluateSingleExt(gkr_index, layer_index, 3);
    for (var i = 0; i < combined_points_size; i++) {
        const_mle.points[i] <== combined_points[i];
        mul_mle.points[i] <== combined_points[i];
        forward_x_mle.points[i] <== combined_points[i];
        forward_y_mle.points[i] <== combined_points[i];
    }

    const_out <== const_mle.out;
    mul_out <== mul_mle.out;
    forward_x_out <== forward_x_mle.out;
    forward_y_out <== forward_y_mle.out;
}
