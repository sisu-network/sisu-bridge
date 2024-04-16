use ark_ff::Field;

pub struct CircuitUtil {}

impl CircuitUtil {
    pub fn modulo<F: Field>() -> F {
        F::from(79228162514264337593543950336u128) // 2^96
    }

    pub fn mul_labels() -> Vec<String> {
        vec!["div_y_x_2", "mul_y_x_pr", "diff_y_qp"]
            .iter()
            .map(|x| x.to_string())
            .collect()
    }

    pub fn num384_labels() -> Vec<String> {
        vec![
            // mul
            "div_y_x_2",
            "mul_y_x_pr",
            "diff_y_qp",
            // add
            "diff_x_qp",
            "diff_x_pr",
            "div_y_x",
            "div_y_x_2_minus_p_x",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect()
    }

    pub fn cmp_lt_label(left_label: &str, right_label: &str) -> String {
        format!("cmp_{left_label}_{right_label}_")
    }

    pub fn eq_label(left_label: &str, right_label: &str) -> String {
        format!("eq_{left_label}_{right_label}_")
    }

    pub fn diff_label(left_label: &str, right_label: &str) -> String {
        format!("diff_{left_label}_{right_label}_")
    }

    pub fn mul_label(left_label: &str, right_label: &str) -> String {
        format!("mul_{left_label}_{right_label}_")
    }

    pub fn word_label(label: &str) -> String {
        format!("{label}_word")
    }
}
