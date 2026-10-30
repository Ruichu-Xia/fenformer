# Data

Every image is either rendered by this project or comes from a public dataset with a
published license. Test splits and zero-shot benchmarks are never seen in training, and
no hyperparameter was tuned on them.

## Training mixture

| Source | Train | Val | What it is |
| --- | ---: | ---: | --- |
| `render` | 34,000 | 6,000 | Procedural scenes from a CPU rasterizer: 10 piece-mesh families, random board/piece/table materials and lighting, full 360° orbit, roll ±45°, camera elevation 20–75° (30% at 8–20°) |
| `render_low_angle` | 17,000 | 3,000 | Same renderer, elevation 20–45° (60% at 8–20°), random per-piece rotation, bounded shape variation of the procedural pieces |
| ChessReD | 6,479 | 2,192 | Smartphone photos of one physical set, official split (by game) |
| ChessCog | 4,400 | 146 | Blender renders of one set, official split |
| **Total** | **61,879** | **11,338** | |

Positions in both render sets are real game positions sampled from the
[Lichess database](https://database.lichess.org/) (CC0). Each image is letterboxed to 644 × 644 px without cropping;
board framing happens in the model's preprocessing, identically for training and
inference. Board boxes for training images were precomputed with the same code
(`fenformer.localize.board_box`) to avoid running U²-Net-p every epoch.

Piece meshes: procedural (this project), [ChessSight](https://github.com/tchauffi/ChessSight)
(MIT), [A Beautiful Game](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/ABeautifulGame)
(CC BY 4.0, © 2020 ASWF, models by Moeen and Mujtaba Sayed), [Staunton](https://github.com/clarkerubber/Staunton-Pieces)
(MIT), [Poly Haven](https://polyhaven.com/a/chess_set), and CC0 sets by
[Clint Bellanger](https://opengameart.org/content/chess-pieces-0),
[Lyricsz](https://opengameart.org/content/stylized-chess-pieces),
[tunakron](https://opengameart.org/content/3d-chess-pieces) and
[Deplorable Mountaineer](https://deplorablemountaineer.itch.io/chess-board-and-pieces).

## Benchmarks

| Benchmark | Images | Domain | Relation to training |
| --- | ---: | --- | --- |
| `chessred-test` | 2,129 | Smartphone photos, varied angles | Official test split; same physical set as ChessReD train, unseen games |
| `chesscog-test` | 342 | Blender renders | Official test split; same set as ChessCog train |
| `chesscog-transfer` | 29 | Phone photos of a wooden tournament set | **Zero-shot**: physical set never seen |
| `samryan-500` | 500 | Top-down DSLR photos of a vinyl roll-up board, clutter around it | **Zero-shot**: set, camera geometry and framing never seen |

Sources: [ChessReD](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f)
(Masouris & van Gemert, CC BY-NC-SA 4.0),
[ChessCog](https://osf.io/xf3ka/) and its
[transfer sample](https://github.com/georg-wolflein/chesscog) (Wölflein & Arandjelović),
[samryan18/chess-dataset](https://github.com/samryan18/chess-dataset) (MIT).
`scripts/prepare_benchmarks.py` rebuilds the two zero-shot sets from their sources.
