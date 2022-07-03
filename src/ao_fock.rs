use crate::*;

use itertools::iproduct;
use matrix_util::*;
use ndarray::prelude::*;
use rayon::prelude::*;

impl Molecule {
    // Two electron part of Fock matrix (G(D))
    pub fn construct_ao_g(
        &self,
        density: ArrayView2<f64>,
        matrix: Option<Array2<f64>>,
    ) -> Array2<f64> {
        let n_ao = self.get_n_ao();
        let n_sh = self.get_shells().len();

        let mut ao_g = if let Some(m) = matrix {
            assert_eq!(m.dim(), (n_ao, n_ao));
            m
        } else {
            Array2::zeros((n_ao, n_ao))
        };

        let mut chunks: Vec<_> = iproduct!(0..n_sh, 0..n_sh)
            .map(|(i, j)| (i as i32, j as i32))
            .zip(
                split_1e_mat_mut(ao_g.view_mut(), self.get_shells())
                    .into_iter(),
            )
            .collect();

        let dens_chunks: Vec<_> = iproduct!(0..n_sh, 0..n_sh)
            .map(|(i, j)| (i as i32, j as i32))
            .zip(split_1e_mat(density, self.get_shells()).into_iter())
            .collect();

        let eri_func = cint2e!(int2e_sph);
        let opt_func = cint_opt!(int2e_optimizer);

        chunks.par_iter_mut().for_each_init(
            || {
                (
                    vec![0.0; self.get_max_shell_size().pow(4)],
                    vec![0.0; self.get_max_shell_size().pow(4)],
                    self.opt(opt_func),
                )
            },
            |(buf1, buf2, opt), ((i, j), chunk)| {
                for ((k, l), dens_chunk) in dens_chunks.iter() {
                    let coulomb_shls = [*i, *j, *k, *l];
                    let exchange_shls = [*i, *l, *k, *j];

                    self.int(eri_func, buf1, coulomb_shls, Some(opt));
                    self.int(eri_func, buf2, exchange_shls, Some(opt));

                    let mut buf1_view = ArrayViewMut4::from_shape(
                        (
                            self.get_shell(*i as usize).n_ao,
                            self.get_shell(*j as usize).n_ao,
                            self.get_shell(*k as usize).n_ao,
                            self.get_shell(*l as usize).n_ao,
                        )
                            .f(),
                        buf1,
                    )
                    .unwrap();

                    let mut buf2_view = ArrayViewMut4::from_shape(
                        (
                            self.get_shell(*i as usize).n_ao,
                            self.get_shell(*l as usize).n_ao,
                            self.get_shell(*k as usize).n_ao,
                            self.get_shell(*j as usize).n_ao,
                        )
                            .f(),
                        buf2,
                    )
                    .unwrap();
                    buf2_view.swap_axes(1, 3);

                    buf2_view *= 0.5;

                    buf1_view -= &buf2_view;

                    for (d_row, mut sl1) in dens_chunk
                        .axis_iter(Axis(0))
                        .zip(buf1_view.axis_iter_mut(Axis(2)))
                    {
                        for (&d_elem, mut sl2) in
                            d_row.iter().zip(sl1.axis_iter_mut(Axis(2)))
                        {
                            sl2 *= d_elem;
                            *chunk += &sl2;
                        }
                    }
                }
            },
        );

        ao_g
    }

    // Naive approach, precomputing entire n^4 eri matrix
    pub fn construct_ao_g_naive(
        &self,
        density: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_ao = self.get_n_ao();

        let g = self.construct_int2e(
            cint2e!(int2e_sph),
            1,
            cint_opt!(int2e_optimizer),
            true,
            None,
        );

        let ao_g = Array2::from_shape_fn((n_ao, n_ao), |(i, j)| {
            iproduct!(0..n_ao, 0..n_ao)
                .map(|(k, l)| {
                    density[(k, l)]
                        * (g[(i, j, k, l, 0)] - 0.5 * g[(i, l, k, j, 0)])
                })
                .sum()
        });

        ao_g
    }

    pub fn construct_h(&self, h: Option<Array2<f64>>) -> Array2<f64> {
        let n_ao = self.get_n_ao();

        let h = h.and_then(|mut m| {
            assert_eq!(m.dim(), (n_ao, n_ao));
            m.fill(0.0);
            Some(m.into_shape((n_ao, n_ao, 1)).unwrap())
        });

        let h = self.construct_int1e_sym(cint1e!(int1e_kin_sph), 1, h);
        let h = self.construct_int1e_sym(cint1e!(int1e_nuc_sph), 1, Some(h));

        h.into_shape((n_ao, n_ao)).unwrap()
    }
}
