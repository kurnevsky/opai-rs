use decorum::R64;
use indoc::indoc;
use oppai_field::field::{directions, length, to_pos, to_x, to_y, Field, Pos};
use oppai_field::player::Player;
use oppai_field::zobrist::Zobrist;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rand::Rng;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::sync::Arc;
use std::{
  collections::VecDeque,
  ptr,
  sync::atomic::{AtomicPtr, Ordering},
};

const SEED: [u8; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

struct Output {
  policy: Vec<f64>,
  value: f64,
}

trait Model {
  type E;

  fn predict(&self, width: u32, height: u32, features: Vec<f64>) -> Result<(Vec<f64>, f64), Self::E>;
}

trait TrainableModel: Model {
  fn train(&self, width: u32, height: u32, inputs: Vec<Vec<f64>>, outputs: Vec<Output>) -> Result<(), Self::E>;
}

struct PyModel<'a> {
  py: Python<'a>,
  model: PyObject,
}

impl<'a> PyModel<'a> {
  pub fn new(py: Python<'a>) -> PyResult<Self> {
    let locals = [("tf", py.import("tensorflow")?), ("np", py.import("numpy")?)].into_py_dict(py);
    let model: PyObject = py
      .eval("tf.keras.models.load_model('model.tf')", None, Some(&locals))?
      .extract()?;

    Ok(Self { py, model })
  }
}

impl<'a> Model for PyModel<'a> {
  type E = PyErr;

  fn predict(&self, width: u32, height: u32, features: Vec<f64>) -> PyResult<(Vec<f64>, f64)> {
    let locals = [("tf", self.py.import("tensorflow")?), ("np", self.py.import("numpy")?)].into_py_dict(self.py);

    let depth = features.len() as u32 / width / height;

    locals.set_item("width", width)?;
    locals.set_item("height", height)?;
    locals.set_item("depth", depth)?;
    locals.set_item("x", features)?;

    locals.set_item("model", &self.model)?;

    self.py.run(
      indoc!(
        "
      x = np.array(x)
      x = np.reshape(x, (depth, width, height))
      x = np.transpose(x, (1, 2, 0))
      x = np.expand_dims(x, axis = 0)
      y = model.predict(x)
      policy = y[0][0]
      value = y[1][0]
    "
      ),
      None,
      Some(&locals),
    )?;

    let policy = locals.get_item("policy").unwrap().extract()?;
    let value = locals.get_item("value").unwrap().extract()?;

    Ok((policy, value))
  }
}

impl<'a> TrainableModel for PyModel<'a> {
  fn train(&self, width: u32, height: u32, inputs: Vec<Vec<f64>>, outputs: Vec<Output>) -> PyResult<()> {
    let locals = [("tf", self.py.import("tensorflow")?), ("np", self.py.import("numpy")?)].into_py_dict(self.py);

    let depth = inputs[0].len() as u32 / width / height;

    locals.set_item("count", inputs.len())?;
    locals.set_item("width", width)?;
    locals.set_item("height", height)?;
    locals.set_item("depth", depth)?;
    locals.set_item("x", inputs)?;
    locals.set_item("values", outputs.iter().map(|output| output.value).collect::<Vec<_>>())?;
    locals.set_item(
      "policies",
      outputs.into_iter().map(|output| output.policy).collect::<Vec<_>>(),
    )?;

    locals.set_item("model", &self.model)?;

    self.py.run(
      indoc!(
        "
      x = np.array(x)
      x = np.reshape(x, (count, depth, width, height))
      x = np.transpose(x, (0, 2, 3, 1))

      policies = np.array(policies)
      policies = np.reshape(policies, (count, width * height))
      values = np.array(values)

      model.fit(x, {'policy_output': policies, 'value_output': values})
    "
      ),
      None,
      Some(&locals),
    )?;

    Ok(())
  }
}

fn main() -> Result<(), ()> {
  let gil = Python::acquire_gil();
  let py = gil.python();
  main_(py).map_err(|e| e.print_and_set_sys_last_vars(py))
}

fn main_(py: Python) -> PyResult<()> {
  let model = PyModel::new(py)?;

  // model.train(39, 32, vec![vec![0f64; 39 * 32 * 2]], vec![Output { policy: vec![0f64; 39 * 32], value: 1f64 }])?;

  // let Output { policy, value } = model.predict(39, 32, vec![0f64; 39 * 32 * 2])?;

  let mut rng = XorShiftRng::from_seed(SEED);

  let zobrist = Arc::new(Zobrist::new(length(39, 32) * 2, &mut rng));
  let mut field = Field::new(39, 32, zobrist);

  field.put_point(field.to_pos(39 / 2, 32 / 2), Player::Black);

  episode(&mut field, Player::Red, &model, &mut rng);

  println!("{}", field);

  Ok(())
}

pub fn wave2<F: FnMut(Pos, Pos) -> bool>(width: u32, start_pos: Pos, mut cond: F) {
  let mut q = VecDeque::new();
  q.push_back(start_pos);
  while let Some(pos) = q.pop_front() {
    q.extend(directions(width, pos).iter().filter(|&&next_pos| cond(pos, next_pos)))
  }
}

struct DistancesChange {
  changes: Vec<(Pos, u32)>,
}

impl DistancesChange {
  pub fn new() -> Self {
    Self { changes: Vec::new() }
  }

  pub fn add(&mut self, pos: Pos, value: u32) {
    self.changes.push((pos, value));
  }
}

struct Distances {
  width: u32,
  values: Vec<u32>,
  max_distance: u32,
  changes: Vec<DistancesChange>,
}

impl Distances {
  pub fn new(width: u32, height: u32, max_distance: u32) -> Self {
    let len = length(width, height);
    let mut values = vec![u32::max_value(); len];
    let max_pos = to_pos(width, width - 1, height - 1);
    for x in 0..width as Pos + 2 {
      values[x] = 0;
      values[max_pos + 2 + x] = 0;
    }
    for y in 1..=height as Pos {
      values[y * (width as Pos + 2)] = 0;
      values[(y + 1) * (width as Pos + 2) - 1] = 0;
    }
    Self {
      width,
      values,
      max_distance,
      changes: Vec::new(),
    }
  }

  pub fn from_field(field: &Field, max_distance: u32) -> Self {
    let mut distances = Self::new(field.width(), field.height(), max_distance);
    for &pos in field.points_seq() {
      distances.put_point(pos);
    }
    distances
  }

  pub fn put_point(&mut self, pos: Pos) {
    // out of bounds
    let mut change = DistancesChange::new();
    change.add(pos, self.values[pos]);
    self.values[pos] = 0;
    wave2(self.width, pos, |prev_pos, next_pos| {
      let prev_pos_value = self.values[prev_pos];
      let next_pos_value = self.values[next_pos];
      if next_pos_value > prev_pos_value + 1 {
        change.add(next_pos, next_pos_value);
        self.values[next_pos] = prev_pos_value + 1;
        prev_pos_value + 1 < self.max_distance
      } else {
        false
      }
    });
    self.changes.push(change);
  }

  pub fn undo(&mut self) {
    if let Some(change) = self.changes.pop() {
      for (pos, value) in change.changes {
        self.values[pos] = value;
      }
    }
  }

  pub fn max_distance(&self) -> u32 {
    self.max_distance
  }

  pub fn value(&self, pos: Pos) -> u32 {
    self.values[pos]
  }

  pub fn is_close(&self, pos: Pos) -> bool {
    self.value(pos) <= self.max_distance
  }
}

struct MctsNode {
  pos: Pos,
  n: u64, // visits
  p: f64, // prior probability
  q: f64, // action value
  children: Vec<MctsNode>,
}

const TEMPERATHURE: f64 = 1f64;

impl MctsNode {
  pub fn new(pos: Pos, p: f64) -> Self {
    Self {
      pos,
      n: 0,
      p,
      q: 0f64,
      children: Vec::new(),
    }
  }

  pub fn probability(&self) -> f64 {
    (self.n as f64).powf(1f64 / TEMPERATHURE)
  }

  pub fn mcts_value(&self, parent_n: u64) -> f64 {
    self.q + C_PUCT * self.p * (parent_n as f64).sqrt() / (1 + self.n) as f64
  }
}

fn push_features(field: &Field, player: Player, features: &mut Vec<f64>) {
  features.extend(
    (field.min_pos()..=field.max_pos())
      .into_iter()
      .filter(|&pos| !field.cell(pos).is_bad())
      .map(|pos| if field.cell(pos).is_owner(player) { 1f64 } else { 0f64 }),
  );
}

fn field_features(field: &Field, player: Player) -> Vec<f64> {
  // rotation
  let mut features = Vec::with_capacity((field.width() * field.height() * 2) as usize);
  push_features(field, player, &mut features);
  push_features(field, player.next(), &mut features);
  features
}

fn pos_to_coord(width: u32, pos: Pos) -> usize {
  let x = to_x(width, pos);
  let y = to_y(width, pos);
  (y * width + x) as usize
}

fn coord_to_pos(width: u32, coord: usize) -> Pos {
  let x = coord as u32 % width;
  let y = coord as u32 / width;
  to_pos(width, x, y)
}

fn create_children(field: &Field, policy: Vec<f64>, distances: &Distances, children: &mut Vec<MctsNode>) {
  for pos in field.min_pos()..=field.max_pos() {
    let coord = pos_to_coord(field.width(), pos);
    let p = policy[coord];

    if field.cell(pos).is_putting_allowed() && distances.is_close(pos) && p > 0f64 {
      let child = MctsNode::new(pos, p);
      children.push(child);
    }
  }
  // renormalize
  let sum: f64 = children.iter().map(|child| child.p).sum();
  for child in children.iter_mut() {
    child.p /= sum;
  }
}

fn is_game_ended(field: &Field) -> bool {
  field.points_seq().len() > 50
}

const C_PUCT: f64 = 1f64;

fn search<E, M: Model<E = E>>(
  field: &mut Field,
  player: Player,
  distances: &mut Distances,
  node: &mut MctsNode,
  model: &M,
) -> Result<f64, E> {
  if is_game_ended(field) {
    use std::cmp::Ordering;
    return Ok(match field.score(player).cmp(&0) {
      Ordering::Less => 1f64,
      Ordering::Equal => 0f64,
      Ordering::Greater => -1f64,
    });
  }

  if node.children.is_empty() {
    let (policy, value) = model.predict(field.width(), field.height(), field_features(field, player))?; // TODO: random rotation
    create_children(field, policy, distances, &mut node.children);
    Ok(-value)
  } else {
    let node_n = node.n;
    if let Some(next) = node
      .children
      .iter_mut()
      .max_by_key(|child| R64::from(child.mcts_value(node_n)))
    {
      field.put_point(next.pos, player);
      distances.put_point(next.pos);

      let value = search(field, player.next(), distances, next, model)?;

      field.undo();
      distances.undo();

      node.q = (node.q * node.n as f64 + value) / (node.n + 1) as f64;
      node.n += 1;

      Ok(-value)
    } else {
      unreachable!("Can't select a child.");
    }
  }
}

const MCTS_SIMS: u32 = 25; // 1600;

fn select<R: Rng>(mut nodes: Vec<MctsNode>, rng: &mut R) -> MctsNode {
  println!("{:?}", nodes.iter().map(|c| c.probability()).collect::<Vec<_>>());

  let r = rng.gen_range(0f64, nodes.iter().map(|child| child.probability()).sum::<f64>());
  let mut node = nodes.pop().unwrap();
  let mut sum = node.probability();
  while sum < r {
    node = nodes.pop().unwrap();
    sum += node.probability();
  }
  node
}

fn episode<E, M, R>(field: &mut Field, mut player: Player, model: &M, rng: &mut R) -> Result<(), E>
where
  M: Model<E = E>,
  R: Rng,
{
  let mut node = MctsNode::new(0, 0f64);
  let mut distances = Distances::from_field(field, 3);

  while !is_game_ended(field) {
    for _ in 0..MCTS_SIMS {
      search(field, player, &mut distances, &mut node, model)?;
    }
    node = select(node.children, rng);
    field.put_point(node.pos, player);
    distances.put_point(node.pos);

    println!("{}", field);

    player = player.next();
  }

  Ok(())
}
