pragma circom 2.1.7;
include "../../circuit/sisu/merkle_tree.circom";

component main {public [index]} = VerifyMerklePathMimc(64, 18446744069414584321, 2, 2);
