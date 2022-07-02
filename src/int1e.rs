use libcint::*;
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::{Molecule, Shell};

fn split_mat<'a>(
    mut mat: ArrayViewMut3<'a, f64>,
    shells: &[Shell],
) -> Vec<ArrayViewMut3<'a, f64>> {
    let mut parts = Vec::new();

    for sh1 in shells {
        let (mut top, rest) = mat.split_at(Axis(0), sh1.n_ao);
        mat = rest;

        for sh2 in shells {
            let (chunk, right) = top.split_at(Axis(1), sh2.n_ao);
            top = right;

            parts.push(chunk);
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
        let n_sh = self.get_shells().len();

        let mut matrix = Array3::zeros((n_ao, n_ao, n_comp));

        let mut chunks_mat = split_mat(matrix.view_mut(), self.get_shells());

        let mut chunks: Vec<_> = (0..n_sh)
            .flat_map(|i| (0..n_sh).map(move |j| [j as i32, i as i32]))
            .zip(chunks_mat.as_mut_slice().iter_mut())
            .collect();

        chunks.par_iter_mut().for_each_init(
            || vec![0.0; self.get_max_shell_size().pow(2) * n_comp],
            |buf, (shls, chunk)| {
                self.int(&int_func, buf, *shls, None);

                let buf_view =
                    ArrayView3::from_shape(chunk.dim(), &buf).unwrap();

                chunk.assign(&buf_view);
            },
        );

        matrix
    }

    pub fn construct_int1e_sym<
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
        let n_sh = self.get_shells().len();

        let mut matrix = Array3::zeros((n_ao, n_ao, n_comp));

        let mut chunks: Vec<_> =
            split_mat(matrix.view_mut(), self.get_shells())
                .into_iter()
                .map(|c| Some(c))
                .collect();
        let mut chunks_mat =
            ArrayViewMut2::from_shape((n_sh, n_sh), &mut chunks).unwrap();

        let mut diag_chunks = Vec::new();

        for i in 0..n_sh {
            diag_chunks.push((i, chunks_mat[(i, i)].take().unwrap()));
        }

        let mut off_diag_chunks = Vec::new();

        for i in 0..n_sh {
            for j in 0..i {
                let mut upper_chunk = chunks_mat[(j, i)].take().unwrap();
                upper_chunk.swap_axes(0, 1);

                off_diag_chunks.push((
                    [j as i32, i as i32],
                    chunks_mat[(i, j)].take().unwrap(),
                    upper_chunk,
                ));
            }
        }

        diag_chunks.par_iter_mut().for_each_init(
            || vec![0.0; self.get_max_shell_size().pow(2) * n_comp],
            |buf, (shl, chunk)| {
                self.int(&int_func, buf, [*shl as i32; 2], None);

                let buf_view =
                    ArrayView3::from_shape(chunk.dim(), &buf).unwrap();

                chunk.assign(&buf_view);
            },
        );

        off_diag_chunks.par_iter_mut().for_each_init(
            || vec![0.0; self.get_max_shell_size().pow(2) * n_comp],
            |buf, (shls, chunk1, chunk2)| {
                self.int(&int_func, buf, *shls, None);

                let buf_view =
                    ArrayView3::from_shape(chunk1.dim(), &buf).unwrap();

                chunk1.assign(&buf_view);
                chunk2.assign(&buf_view);
            },
        );

        matrix
    }
}
