mod basisparse;

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_parse_basis() {
        let bas = basisparse::get_basis("cc-pvdz");

        assert_eq!(bas["O"].len(), 3);
    }
}
