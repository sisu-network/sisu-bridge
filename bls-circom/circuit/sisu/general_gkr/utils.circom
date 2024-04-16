pragma circom 2.1.7;

include "../num_func.circom";
include "../configs.gen.circom";
include "../sumcheck/utils.circom";

function calc_general__extra_num_vars(gkr_index) {
    return log2(get_general_gkr__num_replicas(gkr_index));
}

function calc_general_gkr__g_num_vars(gkr_index, layer_index) {
    return calc_general__extra_num_vars(gkr_index)
        + get_general_gkr__single_subcircuit_num_vars(gkr_index, layer_index);
}

function calc_general_gkr__x_num_vars(gkr_index, layer_index) {
    return calc_general_gkr__g_num_vars(gkr_index, layer_index + 1);
}

function calc_general_gkr__max_y_num_vars(gkr_index, layer_index) {
    var single_max_y_num_vars = 0;
    var num_rounds = get_general_gkr__num_rounds(gkr_index);
    for (var dst_layer_index = layer_index+1; dst_layer_index < num_rounds+1; dst_layer_index++) {
        var t = get_general_gkr__single_subcircuit_subset_num_vars(gkr_index, layer_index, dst_layer_index);
        if (t > single_max_y_num_vars) {
            single_max_y_num_vars = t;
        }
    }

    return calc_general__extra_num_vars(gkr_index) + single_max_y_num_vars;
}

function calc_general_gkr__max_reverse_subset_num_vars(gkr_index, layer_index) {
    var single_max_reverse_subset_num_vars = 0;
    var num_rounds = get_general_gkr__num_rounds(gkr_index);
    for (var i = 0; i < layer_index - 1; i++) {
        var t = get_general_gkr__single_subcircuit_subset_num_vars(gkr_index, i, layer_index);
        if (t > single_max_reverse_subset_num_vars) {
            single_max_reverse_subset_num_vars = t;
        }
    }

    return calc_general__extra_num_vars(gkr_index) + single_max_reverse_subset_num_vars;
}

function calc_general_gkr__y_num_vars(gkr_index, src_layer_index, dst_layer_index) {
    return calc_general__extra_num_vars(gkr_index)
        + get_general_gkr__single_subcircuit_subset_num_vars(gkr_index, src_layer_index, dst_layer_index);
}

function calc_general_gkr__round_sumcheck_transcript_size(gkr_index, layer_index) {
    var phase_1_transcript_size = calc_sumcheck__transcript_size(gkr_index * 1000 + layer_index*10 + 1);
    var phase_2_transcript_size = calc_sumcheck__transcript_size(gkr_index * 1000 + layer_index*10 + 2);

    return phase_1_transcript_size + phase_2_transcript_size;
}

function calc_general_gkr__round_transcript_size(gkr_index, layer_index) {
    var result = 0;
    result += calc_general_gkr__round_sumcheck_transcript_size(gkr_index, layer_index); // sumcheck
    result += 1;                                                                        // w_u
    result += get_general_gkr__num_rounds(gkr_index) - layer_index;                     // w_v
    result += calc_sumcheck__transcript_size(gkr_index * 1000 + layer_index*10 + 3);    // combining sumcheck 

    return result;
}

function calc_general_gkr__transcript_size(gkr_index) {
    var result = 0;

    result += get_general_gkr__num_non_zero_outputs(gkr_index) * get_general_gkr__num_replicas(gkr_index);

    var n_rounds = get_general_gkr__num_rounds(gkr_index);
    for (var i = 0; i < n_rounds; i++) {
        result += calc_general_gkr__round_transcript_size(gkr_index, i);
    }

    return result;
}

function extract_general_gkr__transcript__output_range(gkr_index, replica_index) {
    var range[2];
    range[0] = 0;
    range[1] = get_general_gkr__num_non_zero_outputs(gkr_index);

    for (var i = 1; i <= replica_index; i++) {
        range[0] = range[1];
        range[1] = range[0] + get_general_gkr__num_non_zero_outputs(gkr_index);
    }

    return range;
}

function extract_general_gkr__transcript__round_range(gkr_index, layer_index) {
    var range[2];

    range[0] = get_general_gkr__num_non_zero_outputs(gkr_index) * get_general_gkr__num_replicas(gkr_index);
    range[1] = range[0] + calc_general_gkr__round_transcript_size(gkr_index, 0);

    for (var i = 1; i <= layer_index; i++) {
        range[0] = range[1];
        range[1] = range[0] + calc_general_gkr__round_transcript_size(gkr_index, i);
    }

    return range;
}

function extract_general_gkr__round_transcript__sumcheck_range(gkr_index, layer_index) {
    var range[2];
    range[0] = 0;
    range[1] = calc_general_gkr__round_sumcheck_transcript_size(gkr_index, layer_index);

    return range;
}

function extract_general_gkr__round_transcript__w_u(transcript, gkr_index, layer_index) {
    return transcript[calc_general_gkr__round_sumcheck_transcript_size(gkr_index, layer_index)];
}

function extract_general_gkr__round_transcript__w_v_range(gkr_index, layer_index) {
    var range[2];
    range[0] = calc_general_gkr__round_sumcheck_transcript_size(gkr_index, layer_index) + 1;
    range[1] = range[0] + get_general_gkr__num_rounds(gkr_index) - layer_index;
    return range;
}

function extract_general_gkr__round_transcript__combining_transcript_size(gkr_index, layer_index) {
    var range[2];
    range[0] = calc_general_gkr__round_sumcheck_transcript_size(gkr_index, layer_index) // sumcheck
              + 1                                                                       // w_u
              + get_general_gkr__num_rounds(gkr_index) - layer_index;                   // w_v
    range[1] = range[0] + calc_sumcheck__transcript_size(gkr_index * 1000 + layer_index * 10 + 3);
    return range;
}

function extract_general_gkr__round_sumcheck_transcript__phase_1_transcript_size(gkr_index, layer_index) {
    var range[2];
    range[0] = 0;
    range[1] = calc_sumcheck__transcript_size(gkr_index * 1000 + layer_index * 10 + 1);
    return range;
}

function extract_general_gkr__round_sumcheck_transcript__phase_2_transcript_size(gkr_index, layer_index) {
    var range[2];
    range[0] = calc_sumcheck__transcript_size(gkr_index * 1000 + layer_index * 10 + 1);
    range[1] = range[0] + calc_sumcheck__transcript_size(gkr_index * 1000 + layer_index * 10 + 2);
    return range;
}

