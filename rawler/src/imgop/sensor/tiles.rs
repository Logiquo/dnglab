use crate::{
  CFA,
  imgop::{Dim2, Rect},
  pixarray::{Pix2DView, PixF32, RgbF32},
};
use std::marker::PhantomData;

pub struct DemosaicTiler<'a> {
  raw: Pix2DView<'a, f32>,
  cfa: &'a CFA,
  rgb: &'a mut RgbF32,

  size: Dim2,
  halo: Dim2,

  trow: usize,
  tcol: usize,
}

impl<'a> DemosaicTiler<'a> {
  pub fn new(raw: &'a PixF32, cfa: &'a CFA, rgb: &'a mut RgbF32, size: Dim2, halo: Dim2) -> Self {
    Self::from_view(raw.view(raw.rect()), cfa, rgb, size, halo)
  }

  pub fn from_view(raw: Pix2DView<'a, f32>, cfa: &'a CFA, rgb: &'a mut RgbF32, size: Dim2, halo: Dim2) -> Self {
    Self {
      raw,
      cfa,
      rgb,
      size,
      halo,
      trow: 0,
      tcol: 0,
    }
  }
}

impl<'a> Iterator for DemosaicTiler<'a> {
  type Item = Tile<'a>;

  fn next(&mut self) -> Option<Self::Item> {
    let stride_x = self.size.w - self.halo.w * 2;
    let stride_y = self.size.h - self.halo.h * 2;
    let tile_x = self.tcol * stride_x;
    let tile_y = self.trow * stride_y;
    if tile_y >= self.raw.height() {
      return None;
    }

    let tile_width = (self.raw.width() - tile_x).min(self.size.w);
    let tile_height = (self.raw.height() - tile_y).min(self.size.h);
    let tile = Rect::new(crate::imgop::Point::new(tile_x, tile_y), Dim2::new(tile_width, tile_height));

    let core_left = if tile_x == 0 { 0 } else { self.halo.w };
    let core_top = if tile_y == 0 { 0 } else { self.halo.h };
    let core_right = if tile_x + tile_width == self.raw.width() { 0 } else { self.halo.w };
    let core_bottom = if tile_y + tile_height == self.raw.height() { 0 } else { self.halo.h };
    let core = Rect::new(
      crate::imgop::Point::new(tile_x + core_left, tile_y + core_top),
      Dim2::new(tile_width - core_left - core_right, tile_height - core_top - core_bottom),
    );

    if tile_x + stride_x >= self.raw.width() {
      self.tcol = 0;
      self.trow += 1;
    } else {
      self.tcol += 1;
    }

    Some(Tile {
      raw: Pix2DView {
        rect: self.raw.rect,
        inner: self.raw.inner,
      },
      cfa: self.cfa,
      rgb: self.rgb as *mut RgbF32,
      tile,
      core,
      _garbage: [0.0; 3],
      _phantom: PhantomData,
    })
  }
}

pub struct Tile<'a> {
  raw: Pix2DView<'a, f32>,
  cfa: &'a CFA,
  rgb: *mut RgbF32,

  tile: Rect,
  core: Rect,

  _garbage: [f32; 3],
  _phantom: PhantomData<&'a mut RgbF32>,
}

// `DemosaicTiler::next` creates only disjoint core rectangles, and `rgb_mut`
// writes through `rgb` exclusively for pixels in a tile's core.
unsafe impl Send for Tile<'_> {}

impl<'a> Tile<'a> {
  /// Returns the tile-local raw view. `tile` is expressed relative to `raw`.
  #[inline(always)]
  pub fn raw(&self) -> Pix2DView<'a, f32> {
    self.raw.view(self.tile)
  }

  /// Returns a CFA whose `(0, 0)` phase matches the tile-local raw view.
  #[inline(always)]
  pub fn cfa(&self) -> CFA {
    self.cfa.shift(self.tile.p.x, self.tile.p.y)
  }

  /// Returns the output pixel at tile-local `(row, col)` when it belongs to
  /// this tile's core; writes outside the core go to `_garbage`.
  #[inline(always)]
  pub fn rgb_mut(&mut self, row: usize, col: usize) -> &mut [f32; 3] {
    debug_assert!(row < self.tile.d.h, "row is outside the tile");
    debug_assert!(col < self.tile.d.w, "column is outside the tile");

    let raw_x = self.tile.p.x + col;
    let raw_y = self.tile.p.y + row;
    let is_core = raw_x >= self.core.p.x && raw_x < self.core.p.x + self.core.d.w && raw_y >= self.core.p.y && raw_y < self.core.p.y + self.core.d.h;

    if is_core {
      // SAFETY: DemosaicTiler yields pairwise-disjoint cores. Tile's fields
      // are private, so this is the only route to the RGB destination.
      unsafe { (&mut *self.rgb).at_mut(raw_y, raw_x) }
    } else {
      &mut self._garbage
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn raw(width: usize, height: usize) -> PixF32 {
    PixF32::new_with((0..height).flat_map(|y| (0..width).map(move |x| (y * 100 + x) as f32)).collect(), width, height)
  }

  fn assert_tiling(width: usize, height: usize, size: Dim2, halo: Dim2, expected_count: usize) {
    let raw = raw(width, height);
    let cfa = CFA::new("RGGB");
    let mut rgb = RgbF32::new(width, height);
    let mut coverage = vec![0_u8; width * height];

    {
      let tiler = DemosaicTiler::new(&raw, &cfa, &mut rgb, size, halo);
      let tiles: Vec<_> = tiler.collect();
      assert_eq!(tiles.len(), expected_count);

      for tile in &tiles {
        assert!(tile.tile.p.x + tile.tile.d.w <= width);
        assert!(tile.tile.p.y + tile.tile.d.h <= height);
        assert!(tile.core.p.x >= tile.tile.p.x);
        assert!(tile.core.p.y >= tile.tile.p.y);
        assert!(tile.core.p.x + tile.core.d.w <= tile.tile.p.x + tile.tile.d.w);
        assert!(tile.core.p.y + tile.core.d.h <= tile.tile.p.y + tile.tile.d.h);

        let tile_raw = tile.raw();
        for row in 0..tile_raw.height() {
          for col in 0..tile_raw.width() {
            assert_eq!(*tile_raw.at(row, col), raw[(tile.tile.p.y + row) * width + tile.tile.p.x + col]);
          }
        }

        for row in tile.core.p.y..tile.core.p.y + tile.core.d.h {
          for col in tile.core.p.x..tile.core.p.x + tile.core.d.w {
            coverage[row * width + col] += 1;
          }
        }
      }
    }

    assert!(coverage.iter().all(|&count| count == 1));
  }

  #[test]
  fn tiles_even_dimensions_and_even_tile_size() {
    assert_tiling(8, 6, Dim2::new(4, 4), Dim2::new(1, 1), 12);
  }

  #[test]
  fn tiles_odd_dimensions_and_odd_tile_size() {
    assert_tiling(8, 7, Dim2::new(5, 5), Dim2::new(1, 1), 9);
  }

  #[test]
  fn tile_cores_are_pairwise_disjoint() {
    let raw = raw(17, 13);
    let cfa = CFA::new("RGGB");
    let mut rgb = RgbF32::new(17, 13);
    let tiles: Vec<_> = DemosaicTiler::new(&raw, &cfa, &mut rgb, Dim2::new(6, 5), Dim2::new(1, 1)).collect();

    for (index, tile) in tiles.iter().enumerate() {
      for other in tiles.iter().skip(index + 1) {
        assert!(
          tile.core.intersection(&other.core).is_empty(),
          "overlapping cores: {:?} and {:?}",
          tile.core,
          other.core
        );
      }
    }
  }

  fn assert_cfa_is_tile_local(size: Dim2) {
    let raw = raw(8, 7);
    let cfa = CFA::new("RGGB");
    let mut rgb = RgbF32::new(8, 7);

    let tiler = DemosaicTiler::new(&raw, &cfa, &mut rgb, size, Dim2::new(1, 1));
    for tile in tiler {
      let tile_cfa = tile.cfa();
      for row in 0..tile.tile.d.h {
        for col in 0..tile.tile.d.w {
          assert_eq!(tile_cfa.color_at(row, col), cfa.color_at(tile.tile.p.y + row, tile.tile.p.x + col));
        }
      }
    }
  }

  #[test]
  fn cfa_phase_is_correct_for_even_tile_stride() {
    assert_cfa_is_tile_local(Dim2::new(4, 4));
  }

  #[test]
  fn cfa_phase_is_correct_for_odd_tile_stride() {
    assert_cfa_is_tile_local(Dim2::new(5, 5));
  }

  #[test]
  fn rgb_mut_writes_only_the_core() {
    let raw = raw(6, 6);
    let cfa = CFA::new("RGGB");
    let mut rgb = RgbF32::new(6, 6);

    {
      let mut tiler = DemosaicTiler::new(&raw, &cfa, &mut rgb, Dim2::new(4, 4), Dim2::new(1, 1));
      let mut first = tiler.next().unwrap();
      *first.rgb_mut(0, 0) = [1.0, 2.0, 3.0];
      *first.rgb_mut(3, 3) = [9.0, 9.0, 9.0];
      assert_eq!(first._garbage, [9.0, 9.0, 9.0]);

      let mut second = tiler.next().unwrap();
      *second.rgb_mut(1, 1) = [4.0, 5.0, 6.0];
      *second.rgb_mut(0, 1) = [8.0, 8.0, 8.0];
      assert_eq!(second._garbage, [8.0, 8.0, 8.0]);
    }

    assert_eq!(*rgb.at(0, 0), [1.0, 2.0, 3.0]);
    assert_eq!(*rgb.at(1, 3), [4.0, 5.0, 6.0]);
    assert_eq!(*rgb.at(3, 3), [0.0, 0.0, 0.0]);
    assert_eq!(*rgb.at(1, 2), [0.0, 0.0, 0.0]);
  }
}
