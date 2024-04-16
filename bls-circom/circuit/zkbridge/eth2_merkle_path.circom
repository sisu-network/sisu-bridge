pragma circom 2.1.7;

include "../sisu/general_gkr/general_gkr.circom";

template VerifyMerklePath() {
    var gkr_index_layer_0 = get_merkle_path__layer_0__gkr_index();

    assert(get_general_gkr__num_non_zero_outputs(gkr_index_layer_0) == 8);
    var num_sha256 = get_general_gkr__num_replicas(gkr_index_layer_0);
    var gkr_transcript_size = calc_general_gkr__transcript_size(gkr_index_layer_0);

    signal input fiat_shamir_seed;
	signal input gkr_transcript[gkr_transcript_size];

    component verify_gkr = VerifyGeneralGKRTranscript(circuit_index);
    verify_gkr.fiat_shamir_seed <== fiat_shamir_seed;
    verify_gkr.fiat_shamir_in_r <== 0;
    verify_gkr.fiat_shamir_in_round <== 0;
    for (var i = 0; i < gkr_transcript_size; i++) {
        verify_gkr.transcript[i] <== gkr_transcript[i];
    }

    // CIRCUIT OUTPUT FORMAT: W0(0-1)  W1(2-3)  H_IN(4-5)  H_OUT(6-7)

    // SHA1_H_IN === INIT_H_VALUE.
    //
    // Iterate over even sha256 circuit, ensure that H_IN == INIT_H.
    for (var i = 0; i < num_sha256; i += 2) {
        verify_gkr.circuit_output[i][4] == 140949571397636515175436104201107666234;
        verify_gkr.circuit_output[i][5] == 107741833082138526902889282413787139353;
    }

    // SHA2_H_IN === SHA1_H_OUT
    //
    // Iterate over all sha256 circuit, ensure that circuit[i].H_OUT == circuit[i+1].H_IN
    for (var i = 0; i < num_sha256; i += 2) {
        verify_gkr.circuit_output[i][6] == verify_gkr.circuit_output[i+1][4];
        verify_gkr.circuit_output[i][7] == verify_gkr.circuit_output[i+1][5];
    }

    // SHA2_W ==== PADDING W
    //
    // Iterate over odd circuit, ensure that W === W_PADDING
    for (var i = 1; i < num_sha256; i += 2) {
        verify_gkr.circuit_output[i][0] == 170141183460469231731687303715884105728;
        verify_gkr.circuit_output[i][1] == 0;
        verify_gkr.circuit_output[i][2] == 0;
        verify_gkr.circuit_output[i][3] == 512;
    }

    // SHA2_H_OUT === NEXT_SHA1_W0 or NEXT_SHA1_W1
    for (var i = 1; i < num_sha256; i += 2) {

    }
}
