use std::io::Write;
use std::{collections::HashMap, fs::OpenOptions, path::Path};

use ark_ff::Field;

use crate::common::{convert_field_to_string, convert_vec_field_to_string};

#[derive(Clone)]
pub enum FuncReturnValue<F: Field> {
    Number(usize),
    NumberArray(Vec<usize>),
    Field(F),
    FieldArray(Vec<F>),
    String(String),
}

impl<F: Field> FuncReturnValue<F> {
    pub fn to_string(&self) -> String {
        match &self {
            Self::Number(v) => v.to_string(),
            Self::NumberArray(array) => {
                if array.len() == 0 {
                    // NOTE: A bug of circom happens if we returns a
                    // single-elemenent array in the first branch of a function.
                    //
                    // It leads to the wrong values of later "return" commands.
                    //
                    // So we need to return two-element array here.
                    // https://github.com/iden3/circom/issues/245
                    return String::from("[0, 0]");
                }

                if array.len() == 1 {
                    println!("Currently, one-element array {:?} in return statement leads to a wrong behavior in circom", array);
                }

                let mut s = String::from("[");

                for (i, v) in array.iter().enumerate() {
                    if i > 0 {
                        s += ", ";
                    }
                    s += &v.to_string();
                }

                s += "]";
                s
            }
            Self::String(s) => s.clone(),
            Self::Field(f) => convert_field_to_string(f),
            Self::FieldArray(fa) => {
                if fa.len() == 0 {
                    // NOTE: A bug of circom happens if we returns a
                    // single-elemenent array in the first branch of a function.
                    //
                    // It leads to the wrong values of later "return" commands.
                    //
                    // So we need to return two-element array here.
                    // https://github.com/iden3/circom/issues/245
                    return String::from("[0, 0]");
                }

                if fa.len() == 1 {
                    println!("Currently, one-element array {:?} in return statement leads to a wrong behavior in circom", convert_vec_field_to_string(&fa));
                }

                let mut s = String::from("[");

                for (i, v) in fa.iter().enumerate() {
                    if i > 0 {
                        s += ", ";
                    }
                    s += &convert_field_to_string(v);
                }

                s += "]";
                s
            }
        }
    }

    pub fn is_number(&self) -> bool {
        match &self {
            Self::Number(_) => true,
            _ => false,
        }
    }

    pub fn is_number_array(&self) -> bool {
        match &self {
            Self::NumberArray(_) => true,
            _ => false,
        }
    }

    pub fn is_string(&self) -> bool {
        match &self {
            Self::String(_) => true,
            _ => false,
        }
    }

    pub fn is_field(&self) -> bool {
        match &self {
            Self::Field(_) => true,
            _ => false,
        }
    }

    pub fn is_field_array(&self) -> bool {
        match &self {
            Self::FieldArray(_) => true,
            _ => false,
        }
    }
}

#[derive(Clone)]
pub struct FuncGenerator<F: Field> {
    name: String,
    param_names: Vec<String>,
    value_map: HashMap<Vec<usize>, FuncReturnValue<F>>,
    final_return: FuncReturnValue<F>,
}

impl<F: Field> FuncGenerator<F> {
    pub fn new(name: &str, param_names: Vec<&str>) -> Self {
        let param_names = param_names.into_iter().map(|s| s.to_string()).collect();

        Self {
            name: name.to_string(),
            param_names,
            value_map: HashMap::default(),
            final_return: FuncReturnValue::Number(0),
        }
    }

    pub fn custom_final_return(&mut self, final_return_value: FuncReturnValue<F>) {
        self.final_return = final_return_value;
    }

    pub fn add_number(&mut self, param: Vec<usize>, value: usize) {
        if self.value_map.len() == 0 {
            self.final_return = FuncReturnValue::Number(0);
        } else {
            assert!(self.final_return.is_number());
        }

        assert_eq!(self.param_names.len(), param.len());
        assert!(!self.value_map.contains_key(&param));
        self.value_map.insert(param, FuncReturnValue::Number(value));
    }

    pub fn add_number_array(&mut self, param: Vec<usize>, value: Vec<usize>) {
        if self.value_map.len() == 0 {
            self.final_return = FuncReturnValue::NumberArray(vec![0, 0]);
        } else {
            assert!(self.final_return.is_number_array());
        }

        assert_eq!(self.param_names.len(), param.len());
        assert!(!self.value_map.contains_key(&param));
        self.value_map
            .insert(param, FuncReturnValue::NumberArray(value));
    }

    pub fn add_string(&mut self, param: Vec<usize>, value: &str) {
        assert_eq!(self.param_names.len(), param.len());
        assert!(!self.value_map.contains_key(&param));
        self.value_map
            .insert(param, FuncReturnValue::String(value.to_string()));
    }

    pub fn add_field(&mut self, param: Vec<usize>, value: F) {
        if self.value_map.len() == 0 {
            self.final_return = FuncReturnValue::Field(F::ZERO);
        } else {
            assert!(self.final_return.is_field());
        }

        assert_eq!(self.param_names.len(), param.len());
        assert!(!self.value_map.contains_key(&param));
        self.value_map.insert(param, FuncReturnValue::Field(value));
    }

    pub fn add_field_array(&mut self, param: Vec<usize>, value: Vec<F>) {
        if self.value_map.len() == 0 {
            self.final_return = FuncReturnValue::FieldArray(vec![F::ZERO, F::ZERO]);
        } else {
            assert!(self.final_return.is_field());
        }

        assert_eq!(self.param_names.len(), param.len());
        assert!(!self.value_map.contains_key(&param));
        self.value_map
            .insert(param, FuncReturnValue::FieldArray(value));
    }

    pub fn extend(&mut self, other: Self) {
        assert_eq!(self.name, other.name);
        assert_eq!(self.param_names.len(), other.param_names.len());
        for i in 0..self.param_names.len() {
            assert_eq!(self.param_names[i], other.param_names[i]);
        }

        for (k, v) in other.value_map {
            self.value_map.insert(k, v);
        }
    }

    pub fn to_string(&self) -> String {
        let mut content = String::default();

        // function funcname(a, b, c) {
        content += &format!("function {}(", self.name);

        for i in 0..self.param_names.len() {
            if i != 0 {
                content += ", ";
            }

            content += &self.param_names[i];
        }

        content += ") {\n";

        // body
        for (param_value, value) in self.value_map.iter() {
            if self.param_names.len() > 0 {
                content += "    if ("; // BEGIN IF

                for param_index in 0..self.param_names.len() {
                    if param_index != 0 {
                        content += " && "
                    }

                    content += &format!(
                        "{} == {}",
                        self.param_names[param_index], param_value[param_index]
                    );
                }

                content += ") {\n";
            }

            // IF BODY
            content += &format!("        return {};\n", value.to_string());

            if self.param_names.len() > 0 {
                content += "    }\n"; // END IF
            }
        }

        // Panic if not in existing values.
        content += &format!(
            "\n    log(\"failed to get value of function {} at\"",
            self.name
        );

        for i in 0..self.param_names.len() {
            content += &format!(", \" {}=\", {}", self.param_names[i], self.param_names[i]);
        }

        content += ");\n"; // END LOG

        content += "    assert(0 != 0);\n";
        content += &format!("    return {};\n", self.final_return.to_string());

        // }
        content += "}"; // END FUNC

        content
    }
}

pub struct CustomMLEGenerator {
    template_map: HashMap<(usize, usize, usize), String>,
}

impl CustomMLEGenerator {
    pub fn new() -> Self {
        Self {
            template_map: HashMap::default(),
        }
    }

    pub fn add_mle(&mut self, params: (usize, usize, usize), template_name: &str) {
        self.template_map.insert(params, template_name.to_string());
    }

    pub fn to_string(&self) -> String {
        let mut content = String::default();

        // template TemplateName(a, b, c) {
        content += "template CustomMLEEvaluate(gkr_index, layer_index, ext_index) {\n";

        if self.template_map.len() > 0 {
            content +=
            "    var point_size = get_custom_mle__point_size(gkr_index, layer_index, ext_index);\n";
        } else {
            content += "    var point_size = 0;\n";
        }
        content += "    signal input points[point_size];\n";
        content += "    signal output out;\n";

        for (i, (p, template_name)) in self.template_map.iter().enumerate() {
            content += &format!(
                "    if (gkr_index=={} && layer_index=={} && ext_index=={}) {{\n",
                p.0, p.1, p.2
            ); // BEGIN IF

            let component_name = &format!("component_{}", i);

            // IF BODY
            content += &format!(
                "        component {} = {};\n",
                component_name, template_name
            );

            content += &format!("        for (var i = 0; i < point_size; i++) {{\n");
            content += &format!("            {}.points[i] <== points[i];\n", component_name);
            content += &format!("        }}\n");
            content += &format!("        out <== {}.out;\n", component_name);

            content += &format!("    }}\n"); // END IF
        }

        content += "}\n"; // END TEMPLATE

        content
    }

    pub fn extend(&mut self, other: Self) {
        for (k, v) in other.template_map {
            self.template_map.insert(k, v);
        }
    }
}

pub struct FileGenerator<F: Field> {
    filename: String,
    includes: Vec<String>,
    mle: Option<CustomMLEGenerator>,
    functions: HashMap<String, FuncGenerator<F>>,
}

impl<F: Field> FileGenerator<F> {
    pub fn new(filename: &str) -> Self {
        Self {
            filename: filename.to_string(),
            includes: vec![],
            functions: HashMap::default(),
            mle: None,
        }
    }

    pub fn add_func(&mut self, func: FuncGenerator<F>) {
        if !self.functions.contains_key(&func.name) {
            self.functions.insert(func.name.clone(), func);
        } else {
            let f = self.functions.get_mut(&func.name).unwrap();
            f.extend(func);
        }
    }

    pub fn extend_funcs(&mut self, funcs: Vec<FuncGenerator<F>>) {
        for func in funcs {
            self.add_func(func);
        }
    }

    pub fn init_mle(&mut self) {
        assert!(self.mle.is_none());
        self.mle = Some(CustomMLEGenerator::new());
    }

    pub fn add_mle(&mut self, mle: CustomMLEGenerator) {
        self.mle.as_mut().unwrap().extend(mle);
    }

    pub fn extend_mle(&mut self, mle: Vec<CustomMLEGenerator>) {
        for m in mle {
            self.add_mle(m);
        }
    }

    pub fn include(&mut self, path: &str) {
        self.includes.push(path.to_string());
    }

    pub fn create(self) {
        if Path::new(&self.filename).exists() {
            std::fs::remove_file(&self.filename).unwrap();
        }

        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.filename)
            .unwrap();

        writeln!(f, "// This is GENERATED file. DO NOT EDIT").unwrap();
        writeln!(f, "pragma circom 2.1.7;").unwrap();
        writeln!(f, "").unwrap();

        for path in self.includes {
            writeln!(f, "include {:?};", path).unwrap();
        }
        writeln!(f, "").unwrap();

        for (_, func) in self.functions {
            writeln!(f, "{}", func.to_string()).unwrap();
            writeln!(f, "").unwrap();
        }

        if self.mle.is_some() {
            writeln!(f, "{}", self.mle.as_ref().unwrap().to_string()).unwrap();
            writeln!(f, "").unwrap();
        }
    }
}

#[cfg(test)]
pub mod tests {
    use crate::field::FpSisu;

    use super::FuncGenerator;

    #[test]
    fn test_code_gen() {
        let mut generator =
            FuncGenerator::<FpSisu>::new("get_sumcheck_n_rounds", vec!["gkr_index", "layer_index"]);

        generator.add_number(vec![1, 1], 2);
        generator.add_number(vec![1, 2], 3);
        generator.add_number(vec![2, 1], 4);

        println!("{}", generator.to_string());
    }
}
