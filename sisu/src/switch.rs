#[macro_export]
macro_rules! switch {
    ($v:expr; $($a:expr => $b:expr,)* _ => $e:expr $(,)?) => {
        match $v {
            $(v if v == $a => $b,)*
            _ => $e,
        }
    };
}
