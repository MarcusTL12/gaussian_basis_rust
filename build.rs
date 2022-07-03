use std::{env, fs::create_dir_all, path::Path};

use ao_basis::{
    basisparser::{self, get_basis},
    save_basis,
};

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let top_dir = Path::new(&out_dir).join("ao_basis");
    create_dir_all(top_dir.clone()).unwrap();

    for basis_name in basisparser::basis_names() {
        let basis_dir = top_dir.clone().join(&basis_name);
        create_dir_all(basis_dir.clone()).unwrap();
        let basis = get_basis(&basis_name);

        save_basis(basis_dir.to_str().unwrap(), &basis);
    }
}
