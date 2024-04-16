pub const WORD_COUNT: usize = 4;
pub const WORD_SIZE: usize = 96;

pub const COEFFS_128: [u128; 128] = [
    170141183460469231731687303715884105728,
    85070591730234615865843651857942052864,
    42535295865117307932921825928971026432,
    21267647932558653966460912964485513216,
    10633823966279326983230456482242756608,
    5316911983139663491615228241121378304,
    2658455991569831745807614120560689152,
    1329227995784915872903807060280344576,
    664613997892457936451903530140172288,
    332306998946228968225951765070086144,
    166153499473114484112975882535043072,
    83076749736557242056487941267521536,
    41538374868278621028243970633760768,
    20769187434139310514121985316880384,
    10384593717069655257060992658440192,
    5192296858534827628530496329220096,
    2596148429267413814265248164610048,
    1298074214633706907132624082305024,
    649037107316853453566312041152512,
    324518553658426726783156020576256,
    162259276829213363391578010288128,
    81129638414606681695789005144064,
    40564819207303340847894502572032,
    20282409603651670423947251286016,
    10141204801825835211973625643008,
    5070602400912917605986812821504,
    2535301200456458802993406410752,
    1267650600228229401496703205376,
    633825300114114700748351602688,
    316912650057057350374175801344,
    158456325028528675187087900672,
    79228162514264337593543950336,
    39614081257132168796771975168,
    19807040628566084398385987584,
    9903520314283042199192993792,
    4951760157141521099596496896,
    2475880078570760549798248448,
    1237940039285380274899124224,
    618970019642690137449562112,
    309485009821345068724781056,
    154742504910672534362390528,
    77371252455336267181195264,
    38685626227668133590597632,
    19342813113834066795298816,
    9671406556917033397649408,
    4835703278458516698824704,
    2417851639229258349412352,
    1208925819614629174706176,
    604462909807314587353088,
    302231454903657293676544,
    151115727451828646838272,
    75557863725914323419136,
    37778931862957161709568,
    18889465931478580854784,
    9444732965739290427392,
    4722366482869645213696,
    2361183241434822606848,
    1180591620717411303424,
    590295810358705651712,
    295147905179352825856,
    147573952589676412928,
    73786976294838206464,
    36893488147419103232,
    18446744073709551616,
    9223372036854775808,
    4611686018427387904,
    2305843009213693952,
    1152921504606846976,
    576460752303423488,
    288230376151711744,
    144115188075855872,
    72057594037927936,
    36028797018963968,
    18014398509481984,
    9007199254740992,
    4503599627370496,
    2251799813685248,
    1125899906842624,
    562949953421312,
    281474976710656,
    140737488355328,
    70368744177664,
    35184372088832,
    17592186044416,
    8796093022208,
    4398046511104,
    2199023255552,
    1099511627776,
    549755813888,
    274877906944,
    137438953472,
    68719476736,
    34359738368,
    17179869184,
    8589934592,
    4294967296,
    2147483648,
    1073741824,
    536870912,
    268435456,
    134217728,
    67108864,
    33554432,
    16777216,
    8388608,
    4194304,
    2097152,
    1048576,
    524288,
    262144,
    131072,
    65536,
    32768,
    16384,
    8192,
    4096,
    2048,
    1024,
    512,
    256,
    128,
    64,
    32,
    16,
    8,
    4,
    2,
    1,
];

pub const COEFFS_96: [u128; 96] = [
    39614081257132168796771975168,
    19807040628566084398385987584,
    9903520314283042199192993792,
    4951760157141521099596496896,
    2475880078570760549798248448,
    1237940039285380274899124224,
    618970019642690137449562112,
    309485009821345068724781056,
    154742504910672534362390528,
    77371252455336267181195264,
    38685626227668133590597632,
    19342813113834066795298816,
    9671406556917033397649408,
    4835703278458516698824704,
    2417851639229258349412352,
    1208925819614629174706176,
    604462909807314587353088,
    302231454903657293676544,
    151115727451828646838272,
    75557863725914323419136,
    37778931862957161709568,
    18889465931478580854784,
    9444732965739290427392,
    4722366482869645213696,
    2361183241434822606848,
    1180591620717411303424,
    590295810358705651712,
    295147905179352825856,
    147573952589676412928,
    73786976294838206464,
    36893488147419103232,
    18446744073709551616,
    9223372036854775808,
    4611686018427387904,
    2305843009213693952,
    1152921504606846976,
    576460752303423488,
    288230376151711744,
    144115188075855872,
    72057594037927936,
    36028797018963968,
    18014398509481984,
    9007199254740992,
    4503599627370496,
    2251799813685248,
    1125899906842624,
    562949953421312,
    281474976710656,
    140737488355328,
    70368744177664,
    35184372088832,
    17592186044416,
    8796093022208,
    4398046511104,
    2199023255552,
    1099511627776,
    549755813888,
    274877906944,
    137438953472,
    68719476736,
    34359738368,
    17179869184,
    8589934592,
    4294967296,
    2147483648,
    1073741824,
    536870912,
    268435456,
    134217728,
    67108864,
    33554432,
    16777216,
    8388608,
    4194304,
    2097152,
    1048576,
    524288,
    262144,
    131072,
    65536,
    32768,
    16384,
    8192,
    4096,
    2048,
    1024,
    512,
    256,
    128,
    64,
    32,
    16,
    8,
    4,
    2,
    1,
];

pub const MUL_EXTRA_BIT_COEFFS: [u128; 2] = [
    158456325028528675187087900672u128,
    79228162514264337593543950336u128 * 1,
];

pub const P_381_COEFFS: &[u128; 4] = &[
    8047903782086192180586325942,
    20826981314825584179608359615,
    31935979117156477062286671870,
    54880396502181392957329877675,
];