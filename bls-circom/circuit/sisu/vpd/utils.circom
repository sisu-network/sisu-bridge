pragma circom 2.1.7;

include "../gkr/utils.circom";
include "../fri/utils.circom";
include "../configs.gen.circom";

function calc_vpd__input_size(vpd_index) {
    return log2(get_vpd__num_workers(vpd_index)) + get_vpd__single_input_size(vpd_index);
}

function calc_vpd__first_query__single_query_size(vpd_index) {
    return get_vpd__num_workers(vpd_index) + 1 + get_vpd__first_query_path_size(vpd_index);
}

function calc_vpd__first_query_size(vpd_index) {
    return 2 * calc_vpd__first_query__single_query_size(vpd_index);
}

function calc_vpd__commitment_size() {
    return 1;
}

function calc_fft_gkr_index(vpd_index) {
    return 1000 + vpd_index;
}

function calc_vpd__transcript_size(vpd_index) {
    var query_size = calc_vpd__first_query_size(vpd_index);
    var p_fri_query_transcript_size = calc_fri__query_transcript_size(vpd_index);

    var result = 0;
    result += get_vpd__num_workers(vpd_index); // output
    result += calc_fri__commitment_transcript_size(vpd_index); // fri commitment
    result += 1; // h_root
    result += calc_gkr__transcript_size(calc_fft_gkr_index(vpd_index)); // gkr transcript.

    for (var i = 0; i < get_vpd__num_repetitions(vpd_index); i++) {
        result += query_size + query_size + p_fri_query_transcript_size;
    }

    return result;
}

function extract_vpd__commitment__l_root(transcript) {
    return transcript[0];
}

function extract_vpd__transcript__output_range(vpd_index) {
    var range[2];
    range[0] = 0;
    range[1] = get_vpd__num_workers(vpd_index);

    return range;
}

function extract_vpd__transcript__fri_commitment_range(vpd_index) {
    var range[2];
    range[0] = get_vpd__num_workers(vpd_index);
    range[1] = range[0] + calc_fri__commitment_transcript_size(vpd_index);

    return range;
}

function extract_vpd__transcript__h_root(transcript, vpd_index) {
    return transcript[get_vpd__num_workers(vpd_index) + calc_fri__commitment_transcript_size(vpd_index)];
}

function extract_vpd__transcript__q_gkr_transcript_range(vpd_index) {
    var range[2];
    range[0] = get_vpd__num_workers(vpd_index) + calc_fri__commitment_transcript_size(vpd_index) + 1;
    range[1] = range[0] + calc_gkr__transcript_size(calc_fft_gkr_index(vpd_index));

    return range;
}

function extract_vpd__transcript__l_first_query_range(vpd_index, repetition_index) {
    var query_size = calc_vpd__first_query_size(vpd_index);
    var p_fri_query_transcript_size = calc_fri__query_transcript_size(vpd_index);

    var range[2];
    range[0] = get_vpd__num_workers(vpd_index)
        + calc_fri__commitment_transcript_size(vpd_index)
        + 1
        + calc_gkr__transcript_size(calc_fft_gkr_index(vpd_index));
    
    range[1] = range[0] + query_size;

    for (var i = 1; i <= repetition_index; i++) {
        // ignore h_first_query and p_fri_query_transcript
        range[0] = range[1] + query_size + p_fri_query_transcript_size;
        range[1] = range[0] + query_size;
    }

    return range;
}

function extract_vpd__transcript__h_first_query_range(vpd_index, repetition_index) {
    var query_size = calc_vpd__first_query_size(vpd_index);
    var p_fri_query_transcript_size = calc_fri__query_transcript_size(vpd_index);

    var range[2];
    range[0] = get_vpd__num_workers(vpd_index)
        + calc_fri__commitment_transcript_size(vpd_index)
        + 1
        + calc_gkr__transcript_size(calc_fft_gkr_index(vpd_index))
        + query_size;
    
    range[1] = range[0] + query_size;

    for (var i = 1; i <= repetition_index; i++) {
        // ignore p_fri_query_transcript and next l_first_query.
        range[0] = range[1] + p_fri_query_transcript_size + query_size;
        range[1] = range[0] + query_size;
    }

    return range;
}

function extract_vpd__transcript__fri_transcript_range(vpd_index, repetition_index) {
    var query_size = calc_vpd__first_query_size(vpd_index);
    var p_fri_query_transcript_size = calc_fri__query_transcript_size(vpd_index);

    var range[2];
    range[0] = get_vpd__num_workers(vpd_index)
        + calc_fri__commitment_transcript_size(vpd_index)
        + 1
        + calc_gkr__transcript_size(calc_fft_gkr_index(vpd_index))
        + query_size
        + query_size;
    
    range[1] = range[0] + p_fri_query_transcript_size;

    for (var i = 1; i <= repetition_index; i++) {
        // ignore next l_first_query and next h_first_query.
        range[0] = range[1] + query_size + query_size;
        range[1] = range[0] + p_fri_query_transcript_size;
    }

    return range;
}

function extract_vpd__first_query__single_query_range(vpd_index, query_index) {
    assert(query_index == 0 || query_index == 1);

    var single_query_size = calc_vpd__first_query__single_query_size(vpd_index);

    var range[2];
    range[0] = query_index * single_query_size;
    range[1] = range[0] + single_query_size;

    return range;
}

function extract_vpd__first_single_query__evaluations_range(vpd_index) {
    var range[2];
    range[0] = 0;
    range[1] = get_vpd__num_workers(vpd_index);

    return range;
}

function extract_vpd__first_single_query__evaluations_hash(transcript, vpd_index) {
    return transcript[get_vpd__num_workers(vpd_index)];
}

function extract_vpd__first_single_query__path_range(vpd_index) {
    var range[2];
    range[0] = get_vpd__num_workers(vpd_index) + 1;
    range[1] = range[0] + get_vpd__first_query_path_size(vpd_index);

    return range;
}
