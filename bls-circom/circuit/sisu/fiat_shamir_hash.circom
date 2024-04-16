pragma circom 2.1.7;

include "./fr_bn254.circom";
include "./mimc.circom";

function get_hash_byte_count(n) {
	return  (n + 7) \ 8;
}

function get_max_u64() {
	return 18446744073709551616;
}

template HashFieldToInteger(modulus) {
	signal input in;
	signal output out;

	// u64 = in % 2^64-1.
	var max_u64 = get_max_u64();
	signal u64 <-- in % max_u64;
	signal a <-- in \ max_u64;
	in === a * max_u64 + u64;

	// out = u64 % modulus.
	out <-- u64 % modulus;
	signal b <-- u64 \ modulus;

	u64 === b * modulus + out;
}

template FiatShamirHashOneToOne() {
	signal input in;
	signal input in_r;
	signal input in_round;

	signal output out;
	signal output out_r;
	signal output out_round;

	component mimc = MimcMultiple(3);
	mimc.in[0] <== in_round;
	mimc.in[1] <== in;
	mimc.in[2] <== in_r;

	out <== mimc.out;
	out_r <== mimc.out;
	out_round <== in_round + 1;
}

template FiatShamirHashOneToMany(out_len) {
	signal input in;
	signal input in_r;
	signal input in_round;

	signal output out[out_len];
	signal output out_r;
	signal output out_round;

	component fiat_shamir_one[out_len];
	for (var i = 0; i < out_len; i++) {
		fiat_shamir_one[i] = FiatShamirHashOneToOne();
		if (i == 0) {
			fiat_shamir_one[i].in <== in;
			fiat_shamir_one[i].in_r <== in_r;
			fiat_shamir_one[i].in_round <== in_round;
		} else {
			fiat_shamir_one[i].in <== fiat_shamir_one[i - 1].out;
			fiat_shamir_one[i].in_r <== fiat_shamir_one[i - 1].out_r;
			fiat_shamir_one[i].in_round <== fiat_shamir_one[i - 1].out_round;
		}
	}

	for (var i = 0; i < out_len; i++) {
		out[i] <== fiat_shamir_one[i].out;
	}

	out_round <== fiat_shamir_one[out_len - 1].out_round;
	out_r <== fiat_shamir_one[out_len - 1].out_r;
}

template FiatShamirHashManyToOne(in_len) {
	signal input in[in_len];
	signal input in_r;
	signal input in_round;

	signal output out;
	signal output out_r;
	signal output out_round;

	component mimc = MimcMultiple(in_len);
	for (var i = 0; i < in_len; i++) {
		mimc.in[i] <== in[i];
	}

	component fiat_shamir_hash = FiatShamirHashOneToOne();
	fiat_shamir_hash.in <== mimc.out;
	fiat_shamir_hash.in_r <== in_r;
	fiat_shamir_hash.in_round <== in_round;

	out <== fiat_shamir_hash.out;
	out_r <== fiat_shamir_hash.out_r;
	out_round <== fiat_shamir_hash.out_round;
}

// Hash many to many
template FiatShamirHashManyToMany(in_len, out_len) {
	signal input in[in_len];
	signal input in_r;
	signal input in_round;

	signal output out[out_len];
	signal output out_r;
	signal output out_round;


	// Hash input
	component mimc = MimcMultiple(in_len);
	for (var i = 0; i < in_len; i++) {
		mimc.in[i] <== in[i];
	}

	component fiat_shamir_hash = FiatShamirHashOneToMany(out_len);
	fiat_shamir_hash.in <== mimc.out;
	fiat_shamir_hash.in_r <== in_r;
	fiat_shamir_hash.in_round <== in_round;


	for (var i = 0; i < out_len; i++) {
		out[i] <== fiat_shamir_hash.out[i];
	}

	out_round <== fiat_shamir_hash.out_round;
	out_r <== fiat_shamir_hash.out_r;
}
