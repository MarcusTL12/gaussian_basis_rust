use std::{
    env,
    fs::{copy, create_dir_all, read_dir},
    path::Path,
};

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let basis_dir = Path::new(&out_dir).join("ao_basis");
    create_dir_all(basis_dir.clone()).unwrap();

    for basis_file in read_dir("ao_basis").unwrap() {
        let entry = basis_file.unwrap();
        copy(entry.path(), basis_dir.clone().join(entry.file_name())).unwrap();
    }
}
