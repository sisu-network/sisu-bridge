pragma circom 2.1.7;

include "./hash_to_field.circom";
include "./bls_signature.circom";

template VerifySignature(MSG_LEN, n, k) {
  signal input msg[MSG_LEN];
  signal input pubkey[2][k];
  signal input signature[2][2][k];

  component hash_to_field = HashToField(MSG_LEN);
  hash_to_field.msg <== msg;

  component core_verify = CoreVerifyPubkeyG1(n, k);
  core_verify.pubkey <== pubkey;
  core_verify.signature <== signature;
  core_verify.hash <== hash_to_field.out;
}
