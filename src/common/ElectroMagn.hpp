/* _____________________________________________________________________ */
//! \file ElectroMagn.hpp

//! \brief Structure to store global current and EM fields

/* _____________________________________________________________________ */

#pragma once

#include <cmath>
#include <cstdio>
#include <fstream>

#include "Params.hpp"

//! Structure to store global current and EM fields
class ElectroMagn {
public:
  using view_t = Kokkos::View<double ***>;
  using hostview_t = typename view_t::host_mirror_type;

  ElectroMagn() {}

  //! n nodes on global primal grid (N Cells+1)
  int nx_p_m, ny_p_m, nz_p_m;
  //! n nodes on global dual grid (N Primal+1 | N Cells+2)
  int nx_d_m, ny_d_m, nz_d_m;
  //! Delta in each dimenion | cell size
  double dx_m, dy_m, dz_m, cell_volume_m;
  //! Invert certain value to avoid div
  double inv_dx_m, inv_dy_m, inv_dz_m, inv_cell_volume_m;
  //! min position of the grids (position of the node (0,0,0) of the primal grid)
  //! Position in the primal grid is the same reference as for particles
  double min_x_, min_y_, min_z_;

  /*
    FIELDS on Yee lattice | Staggered grid
    Ex -> (ix+1/2)*dx, iy*dy      , iz*dz       same for Jx
    Ey -> ix*dx      , (iy+1/2)*dy, iz*dz       same for Jy
    Ez -> ix*dx      , iy*dy      , (iz+1/2)*dz same for Jz

    Bx -> ix*dx      , (iy+1/2)*dy, (iz+1/2)*dz
    By -> (ix+1/2)*dx, iy*dy      , (iz+1/2)*dz
    Bz -> (ix+1/2)*dx, (iy+1/2)*dy, iz*dz
  */

  //! Global electric field
  view_t Ex_m;
  hostview_t Ex_h_m;
  view_t Ey_m;
  hostview_t Ey_h_m;
  view_t Ez_m;
  hostview_t Ez_h_m;

  //! Global courant total field
  view_t Jx_m;
  hostview_t Jx_h_m;
  view_t Jy_m;
  hostview_t Jy_h_m;
  view_t Jz_m;
  hostview_t Jz_h_m;

  // Required by the Antenna operator.
  int J_dual_zx_m;
  int J_dual_zy_m;
  int J_dual_zz_m;

  //! Global magnetic field
  view_t Bx_m;
  hostview_t Bx_h_m;
  view_t By_m;
  hostview_t By_h_m;
  view_t Bz_m;
  hostview_t Bz_h_m;

  // ____________________________________________________________________________
  //
  //! \brief Memory allocation for global fields and init parameters
  //! \param params global parameters
  // ____________________________________________________________________________
  void allocate(const Params &params) {
    nx_p_m = params.nx_p;
    ny_p_m = params.ny_p;
    nz_p_m = params.nz_p;
    nx_d_m = params.nx_d;
    ny_d_m = params.ny_d;
    nz_d_m = params.nz_d;

    dx_m          = params.dx;
    dy_m          = params.dy;
    dz_m          = params.dz;
    cell_volume_m = params.cell_volume;

    inv_dx_m          = params.inv_dx;
    inv_dy_m          = params.inv_dy;
    inv_dz_m          = params.inv_dz;
    inv_cell_volume_m = params.inv_cell_volume;

    min_x_ = params.inf_x;
    min_y_ = params.inf_y;
    min_z_ = params.inf_z;

    DEBUG("Start Allocate current arrays");
    Jx_m = view_t("Jx", nx_d_m, ny_p_m + 2, nz_p_m + 2);
    Jx_h_m = Kokkos::create_mirror_view(Jx_m);
    Kokkos::deep_copy(Jx_m, 0.0);

    Jy_m = view_t("Jy", nx_p_m + 2, ny_d_m, nz_p_m + 2);
    Jy_h_m = Kokkos::create_mirror_view(Jy_m);
    Kokkos::deep_copy(Jy_m, 0.0);

    Jz_m = view_t("Jz", nx_p_m + 2, ny_p_m + 2, nz_d_m);
    Jz_h_m = Kokkos::create_mirror_view(Jz_m);
    Kokkos::deep_copy(Jz_h_m, 0.0);

    J_dual_zx_m = 0;
    J_dual_zy_m = 0;
    J_dual_zz_m = 1;
    DEBUG("End Allocate current arrays");

    DEBUG("Start Allocate electric fields");
    Ex_m = view_t("Ex", nx_d_m, ny_p_m, nz_p_m);
    Ex_h_m = Kokkos::create_mirror_view(Ex_m);
    Kokkos::deep_copy(Ex_h_m, params.E0_m[0]);

    Ey_m = view_t("Ey", nx_p_m, ny_d_m, nz_p_m);
    Ey_h_m = Kokkos::create_mirror_view(Ey_m);
    Kokkos::deep_copy(Ey_h_m, params.E0_m[1]);

    Ez_m = view_t("Ez", nx_p_m, ny_p_m, nz_d_m);
    Ez_h_m = Kokkos::create_mirror_view(Ez_m);
    Kokkos::deep_copy(Ez_h_m, params.E0_m[2]);
    DEBUG("End Allocate electric fields");

    DEBUG("Start Allocate magnetic fields");
    Bx_m = view_t("Bx", nx_p_m, ny_d_m, nz_d_m);
    Bx_h_m = Kokkos::create_mirror_view(Bx_m);
    Kokkos::deep_copy(Bx_h_m, params.B0_m[0]);

    By_m = view_t("By", nx_d_m, ny_p_m, nz_d_m);
    By_h_m = Kokkos::create_mirror_view(By_m);
    Kokkos::deep_copy(By_h_m, params.B0_m[1]);

    Bz_m = view_t("Bz", nx_d_m, ny_d_m, nz_p_m);
    Bz_h_m = Kokkos::create_mirror_view(Bz_m);
    Kokkos::deep_copy(Bz_h_m, params.B0_m[2]);
    DEBUG("End Allocate magnetic fields");

    // Load all field to the device
    DEBUG("Start sync fields");
    sync(minipic::host, minipic::device);
    DEBUG("End sync fields");
  }

  // ____________________________________________________________________________
  //
  //! Reset all currents grid
  // ____________________________________________________________________________
  template <class T_space> void reset_currents(const T_space) {
    if constexpr (std::is_same<T_space, minipic::Host>::value) {
      // ---> Host case
      Kokkos::deep_copy(Jx_h_m, 0.f);
      Kokkos::deep_copy(Jy_h_m, 0.f);
      Kokkos::deep_copy(Jz_h_m, 0.f);
    } else if constexpr (std::is_same<T_space, minipic::Device>::value) {
      // ---> Device case
      Kokkos::deep_copy(Jx_m, 0.f);
      Kokkos::deep_copy(Jy_m, 0.f);
      Kokkos::deep_copy(Jz_m, 0.f);
    }
    Kokkos::fence();
  }

  // ____________________________________________________________________________
  //! \brief sync host <-> device
  // ____________________________________________________________________________
  template <class T_from, class T_to> void sync(const T_from, const T_to) {
    static_assert(
      !std::is_same<T_from, T_to>::value,
      "ElectroMagn::sync: Invalid combination of from and to");

    if constexpr (std::is_same<T_to, minipic::Host>::value) {
      // ---> Host case
      Kokkos::deep_copy(Ex_h_m, Ex_m);
      Kokkos::deep_copy(Ey_h_m, Ey_m);
      Kokkos::deep_copy(Ez_h_m, Ez_m);

      Kokkos::deep_copy(Jx_h_m, Jx_m);
      Kokkos::deep_copy(Jy_h_m, Jy_m);
      Kokkos::deep_copy(Jz_h_m, Jz_m);

      Kokkos::deep_copy(Bx_h_m, Bx_m);
      Kokkos::deep_copy(By_h_m, By_m);
      Kokkos::deep_copy(Bz_h_m, Bz_m);
    } else if constexpr (std::is_same<T_to, minipic::Device>::value) {
      // ---> Device case
      Kokkos::deep_copy(Ex_m, Ex_h_m);
      Kokkos::deep_copy(Ey_m, Ey_h_m);
      Kokkos::deep_copy(Ez_m, Ez_h_m);

      Kokkos::deep_copy(Jx_m, Jx_h_m);
      Kokkos::deep_copy(Jy_m, Jy_h_m);
      Kokkos::deep_copy(Jz_m, Jz_h_m);

      Kokkos::deep_copy(Bx_m, Bx_h_m);
      Kokkos::deep_copy(By_m, By_h_m);
      Kokkos::deep_copy(Bz_m, Bz_h_m);
    }
    Kokkos::fence();
  }

  void resize(const int nx, const int ny, const int nz) {
    Kokkos::resize(Ex_h_m, nx, ny, nz);
    Kokkos::resize(Ey_h_m, nx, ny, nz);
    Kokkos::resize(Ez_h_m, nx, ny, nz);

    Kokkos::resize(Jx_h_m, nx, ny, nz);
    Kokkos::resize(Jy_h_m, nx, ny, nz);
    Kokkos::resize(Jz_h_m, nx, ny, nz);

    Kokkos::resize(Bx_h_m, nx, ny, nz);
    Kokkos::resize(By_h_m, nx, ny, nz);
    Kokkos::resize(Bz_h_m, nx, ny, nz);

    Kokkos::resize(Ex_m, nx, ny, nz);
    Kokkos::resize(Ey_m, nx, ny, nz);
    Kokkos::resize(Ez_m, nx, ny, nz);

    Kokkos::resize(Jx_m, nx, ny, nz);
    Kokkos::resize(Jy_m, nx, ny, nz);
    Kokkos::resize(Jz_m, nx, ny, nz);

    Kokkos::resize(Bx_m, nx, ny, nz);
    Kokkos::resize(By_m, nx, ny, nz);
    Kokkos::resize(Bz_m, nx, ny, nz);
  }
};
