# EX16 / GX16 anchor collection

This folder collects sparse supervised pairs for GeoRT. The two devices are
recorded independently and paired by an identical anchor name; they do not
need to be connected at the same time.

## Start the device nodes

EX16 anchors require the existing EX16 publisher:

```bash
python nodes/ex16_zmq_node.py
```

Real GX16 collection requires the existing GX16 node on its default command
endpoint. The collector only sends the read-only `getjs` request:

```bash
python nodes/gx16_zmq_node.py
```

GX16 virtual mode does not require hardware or a GX16 node.

## Collect anchors

Run the collectors in separate terminals:

```bash
conda activate geort
python anchor_collection/collect_ex16_anchors.py
python anchor_collection/collect_gx16_anchors.py
```

Open <http://127.0.0.1:8081> for EX16 and
<http://127.0.0.1:8082> for GX16. Enter the same descriptive name in each
collector, for example `open_01` or `pinch_thumb_index`, then save each side.

EX16 capture uses the median of the latest 0.5 seconds. By default it restores
the calibration in `utils/GeoRT/data/human_ex16_ex16_raw.npz` and caches the
resulting 21 GeoRT human keypoints. If calibration is unavailable, raw EX16
anchors can still be saved but they are not exported as training-ready pairs.

GX16 starts in real read-only mode. Select `Virtual sliders` in the web UI, or
pass `--start-virtual`, to edit an expected pose without hardware. Virtual
sliders never command the physical hand. Self-colliding poses are blocked by
default; saving one requires the explicit collision override checkbox.

## Output

Individual source records are human-readable and recoverable:

```text
anchor_collection/data/ex16/<name>.json
anchor_collection/data/gx16/<name>.json
```

Every save or delete atomically rebuilds
`anchor_collection/data/paired_anchors.npz`. Only same-name pairs with valid
EX16 human keypoints are included. Its arrays are:

- `anchor_names`: `[K]`
- `ex16_qpos_deg`: `[K, 16]`
- `gx16_qpos_deg`: `[K, 16]`
- `gx16_qpos_rad`: `[K, 16]`
- `human_keypoints`: `[K, 21, 3]`, metres

Use `--help` on either script for endpoints, ports, paths, timeout, capture
window and collision-check options.
