use std::cell::RefCell;

use libcint::*;
use ndarray::prelude::*;
use rayon::prelude::*;
use thread_local::ThreadLocal;

use crate::{Molecule, Shell};

fn split_mat<'a>(
    mut mat: ArrayViewMut3<'a, f64>,
    shells: &[Shell],
) -> Vec<([i32; 2], ArrayViewMut3<'a, f64>)> {
    let mut parts = Vec::new();

    for (i, sh1) in shells.iter().enumerate() {
        let (mut top, rest) = mat.split_at(Axis(0), sh1.n_ao);
        mat = rest;

        for (j, sh2) in shells.iter().enumerate() {
            let (chunk, right) = top.split_at(Axis(1), sh2.n_ao);
            top = right;

            parts.push(([j as i32, i as i32], chunk));
        }
    }

    parts
}

impl Molecule {
    pub fn construct_int1e<
        F: Fn(
                &mut [f64],
                [i32; 2],
                &[[i32; 6]],
                &[[i32; 8]],
                &[f64],
                Option<&mut CINToptimizer>,
            ) -> i32
            + std::marker::Sync,
    >(
        &self,
        int_func: F,
        n_comp: usize,
    ) -> Array3<f64> {
        let n_ao = self.get_n_ao();

        let mut matrix = Array3::zeros((n_ao, n_ao, n_comp));

        let mut chunks = split_mat(matrix.view_mut(), self.get_shells());

        let buf_threads = ThreadLocal::new();

        chunks.par_iter_mut().for_each(|(shls, chunk)| {
            let mut buf = buf_threads
                .get_or(|| {
                    RefCell::new(vec![
                        0.0;
                        self.get_max_shell_size().pow(2) * n_comp
                    ])
                })
                .borrow_mut();

            self.int(&int_func, &mut buf, *shls, None);

            let buf_view = ArrayView3::from_shape(chunk.dim(), &buf).unwrap();

            chunk.assign(&buf_view);
        });

        matrix
    }
}
