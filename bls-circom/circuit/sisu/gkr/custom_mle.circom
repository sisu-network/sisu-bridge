pragma circom 2.1.7;

include "../num_func.circom";

template FFT_DivideInverse_ForwardY(factor, num_vars) {
    signal input points[num_vars + 2 * (num_vars+1)];
    signal output out;

    var out_start = 0;
    var in1_start = out_start + num_vars;
    var in2_start = in1_start + num_vars + 1;

    signal results[num_vars+1];
    results[0] <== (1 - points[in1_start+0]) * (1 - points[in2_start+0]);

    signal tmp0[num_vars];
    signal tmp1[num_vars];
    signal tmp2[num_vars];
    signal tmp3[num_vars];


    // out*in1*in2 + (1-out)*(1-in1)*(1-in2)
    for (var i = 0; i < num_vars; i++) {
        var out_index = out_start + i;
        var in1_index = in1_start + i + 1;
        var in2_index = in2_start + i + 1;

        tmp0[i] <== points[in1_index] * points[in2_index];
        tmp1[i] <== (1 - points[in1_index]) * (1 - points[in2_index]);
        tmp2[i] <== points[out_index] * tmp0[i];
        tmp3[i] <== tmp2[i] + (1-points[out_index]) * tmp1[i];

        results[i+1] <== results[i] * tmp3[i];
    }

    out <== results[num_vars] * factor;
}

template FFT_ComputeT_Mul(stage, l) {
    // INPUT:  [t..padding + scalars (2*l)..padding]
    // OUTPUT: [2*t..padding + scalars (2*l)..padding]

    var scalars_num_vars = ilog2_ceil(l) + 1;
    var out_t_num_vars = stage + 1;
    var in_t_num_vars = stage;
    if (stage == 0) {
        in_t_num_vars = 1;
    }

    var in_num_vars = scalars_num_vars + 1;
    if (in_t_num_vars > scalars_num_vars) {
        in_num_vars = in_t_num_vars + 1;
    }

    var out_num_vars = scalars_num_vars + 1;
    if (out_t_num_vars > scalars_num_vars) {
        out_num_vars = out_t_num_vars + 1;
    }

    var out_start = 0;
    var in1_start = out_start + out_num_vars;
    var in2_start = in1_start + in_num_vars;

    signal input points[out_num_vars + 2*in_num_vars];
    signal output out;

    signal result0 <== (1 - points[out_start]) * (1 - points[in1_start]);
    signal result1 <== result0 * points[in2_start];

    signal tmp0 <== points[out_start + out_num_vars - 1] * points[in2_start + in_num_vars-1];
    signal tmp1 <== (1 - points[out_start + out_num_vars - 1]) * (1 - points[in2_start + in_num_vars-1]);
    signal result2 <== result1 * (tmp0 + tmp1);

    signal result3;
    if (stage == 0) {
        result3 <== result2 * (1 - points[in1_start + in_num_vars - 1]);
    } else {
        signal xtmp30[in_t_num_vars];
        signal xtmp31[in_t_num_vars];
        signal xresult3_0[in_t_num_vars];

        for (var i = 0; i < in_t_num_vars; i++) {
            var out_index = out_start + out_num_vars - i - 2;
            var in1_index = in1_start + in_num_vars - i - 1;

            xtmp30[i] <== points[out_index] * points[in1_index];
            xtmp31[i] <== (1 - points[out_index]) * (1 - points[in1_index]);

            if (i == 0) {
                xresult3_0[i] <== result2 * (xtmp30[i] + xtmp31[i]);
            } else {
                xresult3_0[i] <== xresult3_0[i-1] * (xtmp30[i] + xtmp31[i]);
            }
        }

        result3 <== xresult3_0[in_t_num_vars-1];
    }

    var l_log_size = ilog2_ceil(l);
    var stage_binary[32] = dec2bin(stage, l_log_size);

    signal result4[l_log_size];
    for (var i = 0; i < l_log_size; i++) {
        var in2_index = in2_start + in_num_vars - 1 - l_log_size + i;
        if (i == 0) {
            result4[i] <== result3 * (stage_binary[i] * points[in2_index] + (1-stage_binary[i]) * (1-points[in2_index]));
        } else {
            result4[i] <== result4[i-1] * (stage_binary[i] * points[in2_index] + (1-stage_binary[i]) * (1-points[in2_index]));
        }
    }

    signal result5;
    if (in_t_num_vars == scalars_num_vars) {
        result5 <== result4[l_log_size-1];
    } else if (in_t_num_vars < scalars_num_vars) {
        signal xresult5_1[scalars_num_vars - in_t_num_vars];
        for (var i = 0; i < scalars_num_vars - in_t_num_vars; i++) {
            var in1_index = in1_start + in_num_vars - 1 - in_t_num_vars - i;
            if (i == 0) {
                xresult5_1[i] <== result4[l_log_size-1] * (1 - points[in1_index]);
            } else {
                xresult5_1[i] <== xresult5_1[i-1] * (1 - points[in1_index]);
            }
        }
        result5 <== xresult5_1[scalars_num_vars - in_t_num_vars - 1];
    } else {
        signal xresult5_2[in_t_num_vars - scalars_num_vars];
        for (var i = 0; i < in_t_num_vars - scalars_num_vars; i++) {
            var in2_index = in2_start + in_num_vars - 1 - scalars_num_vars - i;
            if (i == 0) {
                xresult5_2[i] <== result4[l_log_size-1] * (1 - points[in2_index]);
            } else {
                xresult5_2[i] <== xresult5_2[i-1] * (1 - points[in2_index]);
            }
        }
        result5 <== xresult5_2[in_t_num_vars - scalars_num_vars - 1];
    }

    signal result6;
    if (out_t_num_vars < scalars_num_vars) {
        signal xresult6[scalars_num_vars - out_t_num_vars];
        for (var i = 0; i < scalars_num_vars - out_t_num_vars; i++) {
            var out_index = out_start + out_num_vars - 1 - out_t_num_vars - i;
            if (i == 0) {
                xresult6[i] <== result5 * (1 - points[out_index]);
            } else {
                xresult6[i] <== xresult6[i-1] * (1 - points[out_index]);
            }
        }

        result6 <== xresult6[scalars_num_vars - out_t_num_vars - 1];
    } else {
        result6 <== result5;
    }

    out <== result6;
}


template FFT_ComputeT_ForwardY(stage, l) {
    // INPUT:  [t..padding + scalars (2*l)..padding]
    // OUTPUT: [2*t..padding + scalars (2*l)..padding]

    var scalars_num_vars = ilog2_ceil(l) + 1;
    var out_t_num_vars = stage + 1;
    var in_t_num_vars = stage;
    if (stage == 0) {
        in_t_num_vars = 1;
    }

    var in_num_vars = scalars_num_vars + 1;
    if (in_t_num_vars > scalars_num_vars) {
        in_num_vars = in_t_num_vars + 1;
    }

    var out_num_vars = scalars_num_vars + 1;
    if (out_t_num_vars > scalars_num_vars) {
        out_num_vars = out_t_num_vars + 1;
    }

    var out_start = 0;
    var in1_start = out_start + out_num_vars;
    var in2_start = in1_start + in_num_vars;

    signal input points[out_num_vars + 2*in_num_vars];
    signal output out;

    signal result1 <== points[out_start] * points[in1_start];
    signal result2 <== result1 * points[in2_start];

    signal result3[scalars_num_vars];
    signal tmp1[scalars_num_vars];
    signal tmp2[scalars_num_vars];
    signal tmp3[scalars_num_vars];
    signal tmp4[scalars_num_vars];
    for (var i = 0; i < scalars_num_vars; i++) {
        var out_index = out_start + out_num_vars - 1 - i;
        var in1_index = in1_start + in_num_vars - 1 - i;
        var in2_index = in2_start + in_num_vars - 1 - i;

        tmp1[i] <== points[in1_index] * points[in2_index];
        tmp2[i] <== (1-points[in1_index]) * (1-points[in2_index]);
        tmp3[i] <== tmp1[i] * points[out_index];
        tmp4[i] <== tmp2[i] * (1-points[out_index]);

        if (i == 0) {
            result3[i] <== result2 * (tmp3[i] + tmp4[i]);
        } else {
            result3[i] <== result3[i-1] * (tmp3[i] + tmp4[i]);
        }
    }

    signal result4;
    if (out_num_vars > scalars_num_vars + 1) {
        signal xresult4[out_num_vars - scalars_num_vars];
        for (var i = 0; i < out_num_vars - scalars_num_vars - 1; i++) {
            var out_index = out_start + i + 1;
            if (i == 0) {
                xresult4[i] <== result3[scalars_num_vars-1] * (1 - points[out_index]);
            } else {
                xresult4[i] <== xresult4[i-1] * (1 - points[out_index]);
            }
        }

        result4 <== xresult4[out_num_vars - scalars_num_vars - 2];
    } else {
        result4 <== result3[scalars_num_vars-1];
    }

    signal result5;
    if (in_num_vars > scalars_num_vars + 1) {
        signal xtmp5[in_num_vars - scalars_num_vars - 1];
        signal xresult5[in_num_vars - scalars_num_vars - 1];
        for (var i = 0; i < in_num_vars - scalars_num_vars - 1; i++) {
            var in1_index = in1_start + i + 1;
            var in2_index = in2_start + i + 1;

            xtmp5[i] <== (1 - points[in1_index]) * (1 - points[in2_index]);

            if (i == 0) {
                xresult5[i] <== result4 * xtmp5[i];
            } else {
                xresult5[i] <== xresult5[i-1] * xtmp5[i];
            }
        }

        result5 <== xresult5[in_num_vars - scalars_num_vars - 2];
    } else {
        result5 <== result4;
    }

    out <== result5;
}

template FFT_Shuffle_ForwardY(num_vars) {
    signal input points[3*num_vars];
    signal output out;

    var out_start = 0;
    var in1_start = out_start + num_vars;
    var in2_start = in1_start + num_vars;

    signal results[num_vars];
    signal tmp0[num_vars];
    signal tmp1[num_vars];
    signal tmp2[num_vars];
    signal tmp3[num_vars];

    for (var i = 0; i < num_vars; i++) {
        var out_index = out_start + i;
        var in1_index = in1_start + num_vars - i - 1;
        var in2_index = in2_start + num_vars - i - 1;

        tmp0[i] <== points[in1_index] * points[in2_index];
        tmp1[i] <== (1 - points[in1_index]) * (1 - points[in2_index]);
        tmp2[i] <== points[out_index] * tmp0[i];
        tmp3[i] <== tmp2[i] + (1-points[out_index]) * tmp1[i];

        if (i == 0) {
            results[i] <== tmp3[i];
        } else {
            results[i] <== results[i-1] * tmp3[i];
        }
    }

    out <== results[num_vars-1];
}

template GetBases(g, log_domain_size) {
    signal output out[log_domain_size];

    out[0] <== g;

    for (var i = 1; i < log_domain_size; i++) {
        out[i] <== out[i-1] * out[i-1];
    }
}

template FFT_Interpolate_ForwardX(g, stage, domain_size) {
    var num_vars = log2(domain_size);

    signal input points[3*num_vars];
    signal output out;

    var out_start = 0;
    var in1_start = out_start + num_vars;
    var in2_start = in1_start + num_vars;

    component bases = GetBases(g, num_vars);

    signal case1[num_vars];
    signal tmp1[num_vars];
    signal tmp1_0[num_vars];
    signal tmp1_1[num_vars];
    for (var i = 0; i < num_vars; i++) {
        var index = num_vars - 1 - i;

        tmp1_0[i] <== (1 - points[in1_start + index]) * (1 - points[in2_start + index]);

        if (i == stage) {
            tmp1[i] <== tmp1_0[i];
        } else {
            tmp1_1[i] <== points[in1_start + index] * points[in2_start + index];

            //   out * tmp1_0 + (1-out) * tmp1_1
            // = out * tmp1_0 + tmp1_1 - out*tmp1_1
            // = out * (tmp1_0 - tmp1_1) + tmp1_1
            tmp1[i] <== points[out_start + index] * (tmp1_0[i] - tmp1_1[i]) + tmp1_1[i];
        }

        if (i == 0) {
            case1[i] <== tmp1[i];
        } else {
            case1[i] <== case1[i-1] * tmp1[i];
        }
    }

    
    signal case2[num_vars];
    signal tmp2[num_vars];
    signal tmp2_0[num_vars];
    signal tmp2_1[num_vars];
    signal tmp2_2[num_vars];
    signal tmp2_3[num_vars];
    for (var i = 0; i < num_vars; i++) {
        var index = num_vars - 1 - i;

        tmp2_0[i] <== points[in1_start + index] * points[in2_start + index];

        if (i == stage) {
            tmp2_1[i] <== points[out_start + index] * bases.out[i] + 1 - points[out_start + index]; 
            tmp2[i] <== tmp2_0[i] * tmp2_1[i];
        } else {
            // out * base
            tmp2_1[i] <== points[out_start + index] * bases.out[i];

            // (1 - in1) * (1 - in2)
            tmp2_2[i] <== (1 - points[in1_start + index]) * (1 - points[in2_start + index]);
            
            // (1 - out) * (1 - in1) * (1 - in2)
            tmp2_3[i] <== (1 - points[out_start + index]) * tmp2_2[i];

            tmp2[i] <== tmp2_1[i] * tmp2_0[i] + tmp2_3[i];
        }

        if (i == 0) {
            case2[i] <== tmp2[i];
        } else {
            case2[i] <== case2[i-1] * tmp2[i];
        }
    }

    out <== case1[num_vars - 1] + case2[num_vars - 1];
}

template FFT_Evaluate_ForwardX_Dummy() {
    signal input points[1];
    signal output out;

    out <== points[0];
}

template FFT_Evaluate_ForwardX(coeffs_size, evaluations_size) {
    var out_num_vars = ilog2_ceil(evaluations_size * 2);
    var in_num_vars = log2(coeffs_size);

    signal input evaluation_index[evaluations_size];
    signal input points[out_num_vars + 2 * in_num_vars];
    signal output out;
}
