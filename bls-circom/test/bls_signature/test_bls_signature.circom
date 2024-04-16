pragma circom 2.1.7;

include "../../circuit/bls_signature.circom";

component main { public [ pubkey, signature, hash ] } = CoreVerifyPubkeyG1(55, 7);
