pragma circom 2.1.7;
include "../../circuit/sisu/configs.gen.circom";

template X() {
    signal input x;

    var a[4] = get_gkr__ext__evaluations(2);
    for (var i = 0; i < 4; i++) {
        log("a[", i, "]", a[i]);
    }
}

component main {public [x]} = X();
