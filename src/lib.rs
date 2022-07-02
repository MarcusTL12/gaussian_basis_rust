mod basisparse;
// mod int1e;
mod molecule;

pub use basisparse::*;
pub use molecule::*;

#[cfg(test)]
mod tests {
    use crate::{basisparse::get_basis, *};

    #[test]
    fn test_parse_basis() {
        let bas = get_basis("cc-pvdz");

        assert_eq!(bas["O"].len(), 3);
    }

    #[test]
    fn test_molecule() {
        use libcint::*;

        let mol = Molecule::new(
            parse_atoms(
                "
    O   0.0     0.0     0.0
    H   1.0     0.0     0.0
    H   0.0     1.0     0.0
",
            ),
            &get_basis("cc-pvdz"),
        );

        let mut buf = [0.0; 9];

        let ovlp_fn = cint1e!(int1e_ovlp_sph);

        mol.int(ovlp_fn, &mut buf, [0, 0], None);

        // From pyscf
        let check = [
            1.0000000000000002,
            -0.21406265175603023,
            0.19438415205281015,
            -0.21406265175603018,
            1.0000000000000002,
            0.7086073285770356,
            0.19438415205281015,
            0.7086073285770356,
            1.0,
        ];

        for (a, b) in buf.iter().zip(check.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }
}
