pub fn rotate(width: u32, height: u32, x: u32, y: u32, rotation: u32) -> (u32, u32) {
  match rotation {
    0 => (x, y),
    1 => (width - x - 1, y),
    2 => (x, height - y - 1),
    3 => (width - x - 1, height - y - 1),
    4 => (y, x),
    5 => (height - y - 1, x),
    6 => (y, width - x - 1),
    7 => (height - y - 1, width - x - 1),
    r => panic!("Invalid rotation number: {}", r),
  }
}

pub fn rotate_back(width: u32, height: u32, x: u32, y: u32, rotation: u32) -> (u32, u32) {
  let back_rotation = match rotation {
    5 => 6,
    6 => 5,
    r => r,
  };
  rotate(width, height, x, y, back_rotation)
}

pub fn rotate_sizes(width: u32, height: u32, rotation: u32) -> (u32, u32) {
  if rotation < 4 {
    (width, height)
  } else {
    (height, width)
  }
}
