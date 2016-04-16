Opai-rs
====

Opai-rs an artificial intelligence for the game of points.

It's written in rust language and implements "points console AI protocol v6". (See /doc/PointsAIProtocol6.txt for details.)

You can play with it using "[missile](https://github.com/kurnevsky/missile)".

Features
====

* Two algorithms for searching the optimal move: UCT, NegaScout (principal variation search).
* UCT caching that persists between moves.
* Trajectories for moves pruning in the NegaScout search tree.
* Lock-free multi-threading for both NegaScout and UCT.
* DFA-based patterns searching.
* DSU to optimize capturing.
* Time-based (`gen_move_with_time`) and complexity-based (`gen_move_with_complexity`) calculations.

Running
====

In order to build opai-rs you need a _nightly_ rust installed on your system.

Rustc version 1.10.0-nightly (2174bd97c 2016-04-14) is known to be able to build opai-rs. You can use "[multirust](https://github.com/brson/multirust)" to specify and update nightly versions of rust.

Once you have rust installed on your system, compile with

```sh
    cargo build --release
```

Run with

```sh
    cargo run --release
```

or with

```sh
    ./target/release/opai-rs
```

License
====

This project is licensed under AGPL version 3 or (at your option) any later version. See LICENSE.txt for details.

Copyright (C) 2015 Kurnevsky Evgeny, Vasya Novikov
