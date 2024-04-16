pragma circom 2.1.7;

include "../configs.gen.circom";
include "../sumcheck/utils.circom";

function calc_gkr_sumcheck_index(gkr_index, layer_index) {
    return gkr_index * 1000 + layer_index;
}

// current num_vars == previous sumcheck__n_rounds.
// Because in GKR, all sumchecks at the same layer have the same rounds, so we
// should have a function to get this value.
function get_gkr__round_current_num_vars(gkr_index, layer_index) {
    assert(layer_index > 0); // the output layer is handled in a different way.
    return get_sumcheck__n_worker_rounds(calc_gkr_sumcheck_index(gkr_index, layer_index - 1));
}

// next num_vars == current sumcheck__n_rounds.
// Because in GKR, all sumchecks at the same layer have the same rounds, so we
// should have a function to get this value.
function get_gkr__round_next_num_vars(gkr_index, layer_index) {
    assert(layer_index > 0); // the output layer is handled in a different way.
    return get_sumcheck__n_worker_rounds(calc_gkr_sumcheck_index(gkr_index, layer_index));
}

function calc_gkr__round_sumcheck_phase_transcript_size(gkr_index, layer_index) {
    return calc_sumcheck__transcript_size(calc_gkr_sumcheck_index(gkr_index, layer_index)); 
}

function calc_gkr__round__sumcheck_transcript_size(gkr_index, layer_index) {
    return 2 * calc_gkr__round_sumcheck_phase_transcript_size(gkr_index, layer_index); 
}

function calc_gkr__round_transcript_size(gkr_index, layer_index) {
    // sumcheck + w_u + w_v.
    return calc_gkr__round__sumcheck_transcript_size(gkr_index, layer_index) + 2; 
}

function calc_gkr__transcript_size(gkr_index) {
    var size = get_gkr__output_size(gkr_index);
    for (var i = 0; i < get_gkr__n_rounds(gkr_index); i++) {
        size += calc_gkr__round_transcript_size(gkr_index, i);
    }

    return size;
}

// Input: transcript of a round gkr (all_sumcheck_transcript).
// Output: transcript of a phase sumcheck.
function extract_gkr__round_sumcheck__phase_range(gkr_index, layer_index, phase) {
    assert(phase == 1 || phase == 2);

    var phase_sumcheck_transcript_size = calc_gkr__round_sumcheck_phase_transcript_size(gkr_index, layer_index);

    var range[2];
    if (phase == 1) {
        range[0] = 0;
    } else {
        range[0] = phase_sumcheck_transcript_size;
    }
    range[1] = range[0] + phase_sumcheck_transcript_size;
    
    return range;
}

// Input: transcript of a round gkr (include all_sumcheck_transcript + w_u + w_v).
// Output: transcript of all sumchecks.
function extract_gkr__round__sumchecks_range(gkr_index, layer_index) {
    var sumcheck_transcript_size = calc_gkr__round__sumcheck_transcript_size(gkr_index, layer_index);

    var range[2];
    range[0] = 0;
    range[1] = range[0] + sumcheck_transcript_size;
    
    return range;
}

// Input: transcript of a round gkr (include all_sumcheck_transcript + w_u + w_v).
// Output: w_u.
function extract_gkr__round__w_u(transcript, gkr_index, layer_index) {
    var sumcheck_transcript_size = calc_gkr__round__sumcheck_transcript_size(gkr_index, layer_index);
    return transcript[sumcheck_transcript_size];
}

// Input: transcript of a round gkr (include all_sumcheck_transcript + w_u + w_v).
// Output: w_v.
function extract_gkr__round__w_v(transcript, gkr_index, layer_index) {
    var sumcheck_transcript_size = calc_gkr__round__sumcheck_transcript_size(gkr_index, layer_index);
    return transcript[sumcheck_transcript_size+1];
}

// Input: transcript of gkr.
// Output: output range.
function extract_gkr__output_range(gkr_index) {
    var range[2];
    range[0] = 0;
    range[1] = get_gkr__output_size(gkr_index);

   return range;
}

// Input: transcript of gkr.
// Output: a round gkr transcript.
function extract_gkr__round_transcript_range(gkr_index, layer_index) {
    var output_size = get_gkr__output_size(gkr_index);

    var range[2];
    range[0] = output_size;
    range[1] = range[0] + calc_gkr__round_transcript_size(gkr_index, 0);

    for (var i = 1; i <= layer_index; i++) {
        range[0] = range[1];
        range[1] = range[0] + calc_gkr__round_transcript_size(gkr_index, i);
    }

    return range;
}