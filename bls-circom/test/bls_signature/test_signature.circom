pragma circom 2.1.7;

include "../../circuit/signature.circom";

component main { public [ pubkey, signature, msg ] } = VerifySignature(32, 55, 7);
