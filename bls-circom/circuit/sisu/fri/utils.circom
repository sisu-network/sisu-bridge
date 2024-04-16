pragma circom  2.1.7;

include "./utils.circom";
include "../num_func.circom";


function calc_fri__commitment_transcript_size(vpd_index) {
    var n_workers = get_fri__n_workers(vpd_index);
    var n_layers = get_fri__n_layers(vpd_index);

    // n_roots = n_layers - 1;
    // n_final_constants = n_workers;
    return n_layers - 1 + n_workers;
}

function calc_fri__query_transcript_size(vpd_index) {
    var n_layers = get_fri__n_layers(vpd_index);
    var result = 0;
    for (var i = 1; i < n_layers; i++) {
        var query_size = get_fri__query_size(vpd_index, i);
        result += query_size * 2;
    }

    return result;
}

function calc_fri__query_path_size(vpd_index, layer_index) {
    assert(layer_index > 0);

    var n_workers = get_fri__n_workers(vpd_index);
    var query_size = get_fri__query_size(vpd_index, layer_index);

    return query_size - 1 - n_workers;
}

function extract_fri__commitment_transcript__root(transcript, vpd_index, layer_index) {
    assert(layer_index > 0);

    return transcript[layer_index - 1];
}

function extract_fri__commitment_transcript__final_constant(transcript, vpd_index, worker_index) {
    var n_layers = get_fri__n_layers(vpd_index);
    return transcript[n_layers - 1 + worker_index];
}


function extract_fri__query_transcript__z_query_range(vpd_index, layer_index) {
    assert(layer_index > 0);
    var query_size = get_fri__query_size(vpd_index, 1);

    var range[2];
    range[0] = 0;
    range[1] = query_size;

    for (var i = 2; i <= layer_index; i++) {
        var old_query_size = get_fri__query_size(vpd_index, i - 1);
        range[0] = range[1] + old_query_size; // bypass op_z_queries.

        // get new query size
        var new_query_size = get_fri__query_size(vpd_index, i);
        range[1] = range[0] + new_query_size;
    }

    return range;
}

function extract_fri__query_transcript__op_z_query_range(vpd_index, layer_index) {
    assert(layer_index > 0);
    var query_size = get_fri__query_size(vpd_index, 1);

    var range[2];
    range[0] = query_size; // bypass z_queries.
    range[1] = range[0] + query_size;

    for (var i = 2; i <= layer_index; i++) {
        var query_size = get_fri__query_size(vpd_index, i);
        range[0] = range[1] + query_size; // bypass z_queries.
        range[1] = range[0] + query_size;
    }

    return range;
}

function extract_fri__query_evaluation_range(vpd_index) {
    var n_workers = get_fri__n_workers(vpd_index);

    var range[2];
    range[0] = 0;
    range[1] = n_workers;

    return range;
}

function extract_fri__query_evaluation_hash(transcript, vpd_index) {
    var n_workers = get_fri__n_workers(vpd_index);

    return transcript[n_workers];
}


function extract_fri__query_path_range(vpd_index, layer_index) {
    var n_workers = get_fri__n_workers(vpd_index);
    var query_size = get_fri__query_size(vpd_index, layer_index);

    var range[2];
    range[0] = n_workers + 1;
    range[1] = query_size;

    return range;
}
