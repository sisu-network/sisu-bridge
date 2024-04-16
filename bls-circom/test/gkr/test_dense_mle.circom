pragma circom 2.1.7;

include "../../circuit/sisu/dense_mle.circom";

// Test for Fp389
// {
//   "evals": ["77", "13", "32", "35"],
//   "points": ["1", "90"]
// }
// component main = DenseEvaluate(9, 389, 4);
component main = DenseEvaluate(64, 18446744069414584321, 16);
