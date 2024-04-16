pub const NUM_LEN: usize = 381;

#[cfg(test)]
pub mod tests {
    fn bytes_to_bits_be(xs: &[u8]) -> Vec<u8> {
        let mut ret = Vec::with_capacity(xs.len() * 8);
        for x in xs {
            let mut tmp = Vec::with_capacity(8);
            for i in 0..8 {
                let mask = 1 << i;
                let bit_is_set = (mask & x) > 0;
                if bit_is_set {
                    tmp.push(1)
                } else {
                    tmp.push(0)
                }
            }
            tmp.reverse();
            ret.extend(tmp)
        }

        ret
    }

    pub fn default_input() -> Vec<Vec<u8>> {
        let p_x: &[u8] = &[
            161, 96, 124, 117, 96, 65, 167, 18, 138, 235, 81, 8, 232, 238, 4, 14, 117, 252, 206,
            105, 196, 247, 11, 82, 247, 147, 42, 11, 216, 1, 33, 88, 253, 254, 89, 51, 132, 73, 63,
            222, 190, 132, 30, 157, 56, 213, 165, 169,
        ];

        let q_x: &[u8] = &[
            177, 201, 69, 155, 77, 224, 174, 66, 37, 148, 8, 65, 4, 16, 11, 172, 105, 17, 171, 47,
            91, 228, 80, 162, 184, 27, 169, 145, 8, 79, 12, 231, 73, 62, 180, 56, 65, 115, 4, 215,
            213, 41, 221, 192, 86, 19, 254, 104,
        ];

        let r_x: &[u8] = &[
            168, 93, 68, 211, 84, 162, 65, 13, 202, 195, 169, 145, 160, 55, 39, 242, 244, 160, 225,
            140, 170, 31, 82, 220, 182, 99, 35, 166, 209, 127, 147, 44, 165, 61, 155, 213, 8, 65,
            59, 24, 219, 58, 110, 112, 73, 228, 174, 210,
        ];

        return vec![
            bytes_to_bits_be(p_x),
            bytes_to_bits_be(q_x),
            bytes_to_bits_be(r_x),
        ];
    }
}
