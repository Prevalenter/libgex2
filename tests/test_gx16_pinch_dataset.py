import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEORT_ROOT = PROJECT_ROOT / "utils" / "GeoRT"
if str(GEORT_ROOT) not in sys.path:
    sys.path.insert(0, str(GEORT_ROOT))

from geort.env.hand import HandKinematicModel  # noqa: E402
from geort.mocap.visualize_robot_kinematics import load_pinch_metadata  # noqa: E402
from geort.trainer import resolve_robot_data_path  # noqa: E402
from geort.utils.mesh_distance import (  # noqa: E402
    CollisionSurface,
    approximate_surface_distance,
)


class MeshDistanceTest(unittest.TestCase):
    def setUp(self):
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        self.surface = CollisionSurface(
            box.vertices, box.faces, coarse_samples=256, fine_samples=2048, seed=3
        )

    def test_separated_boxes_return_surface_gap_and_closest_points(self):
        transform_a = np.eye(4)
        transform_b = np.eye(4)
        transform_b[0, 3] = 1.2
        result = approximate_surface_distance(
            self.surface, transform_a, self.surface, transform_b, fine=True
        )
        self.assertAlmostEqual(result.distance, 0.2, delta=1e-3)
        self.assertAlmostEqual(np.linalg.norm(result.point_a - result.point_b), 0.2, delta=1e-3)

    def test_coarse_and_fine_queries_are_deterministic(self):
        transform_a = np.eye(4)
        transform_b = trimesh.transformations.rotation_matrix(0.3, (0.0, 0.0, 1.0))
        transform_b[0, 3] = 1.4
        first = approximate_surface_distance(
            self.surface, transform_a, self.surface, transform_b, fine=True
        )
        second = approximate_surface_distance(
            self.surface, transform_a, self.surface, transform_b, fine=True
        )
        coarse = approximate_surface_distance(
            self.surface, transform_a, self.surface, transform_b, fine=False
        )
        self.assertAlmostEqual(first.distance, second.distance, places=12)
        self.assertLess(abs(first.distance - coarse.distance), 0.08)


class CollisionPolicyTest(unittest.TestCase):
    def test_allowed_pair_uses_its_own_penetration_limit(self):
        class FakeModel:
            pass

        model = FakeModel()
        model.self_collision_penetrations = lambda _qpos: {
            ("link4", "link8"): 0.009,
            ("link10", "link5"): 0.0004,
        }
        self.assertFalse(
            HandKinematicModel.has_self_collision(
                model,
                np.zeros(16),
                penetration_threshold=0.0005,
                allowed_link_pairs=[("link8", "link4")],
                allowed_penetration_threshold=0.010,
            )
        )
        model.self_collision_penetrations = lambda _qpos: {
            ("link4", "link8"): 0.011,
        }
        self.assertTrue(
            HandKinematicModel.has_self_collision(
                model,
                np.zeros(16),
                penetration_threshold=0.0005,
                allowed_link_pairs=[("link4", "link8")],
                allowed_penetration_threshold=0.010,
            )
        )


class PinchMetadataTest(unittest.TestCase):
    def test_viewer_metadata_round_trip(self):
        count = 4
        pair_count = 3
        metadata = {"distance_range_m": [-0.01, 0.005]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pinch.npz"
            np.savez(
                path,
                primary_pair=np.asarray((0, 1, 2, 0), dtype=np.int8),
                pair_names=np.asarray(("thumb-index", "thumb-middle", "thumb-ring")),
                link_surface_distances_m=np.zeros((count, pair_count)),
                pinch_mask=np.ones((count, pair_count), dtype=bool),
                closest_surface_points_m=np.zeros((count, pair_count, 2, 3)),
                keypoint_distances_m=np.zeros((count, pair_count)),
                pair_penetrations_m=np.zeros((count, pair_count)),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            loaded = load_pinch_metadata(path, count)
        self.assertEqual(loaded["primary_pair"].shape, (count,))
        self.assertEqual(loaded["pair_names"].tolist()[1], "thumb-middle")
        self.assertEqual(loaded["metadata"], metadata)

    def test_regular_dataset_has_no_pinch_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regular.npz"
            np.savez(path, qpos=np.zeros((2, 16)))
            self.assertIsNone(load_pinch_metadata(path, 2))


class PinchConfigTest(unittest.TestCase):
    def test_contact_links_are_physical_terminal_links(self):
        path = GEORT_ROOT / "geort" / "config" / "gx16.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        actual = {
            entry["name"]: entry["contact_link"]
            for entry in config["fingertip_link"]
        }
        self.assertEqual(
            actual,
            {
                "index": "link8",
                "middle": "link12",
                "ring": "link17",
                "thumb": "link4",
            },
        )

    def test_training_robot_dataset_resolves_exact_pinch_file(self):
        expected = (GEORT_ROOT / "data" / "gx16_pinch.npz").resolve()
        self.assertEqual(resolve_robot_data_path("gx16_pinch"), expected)
        self.assertEqual(resolve_robot_data_path("gx16_pinch.npz"), expected)
        with self.assertRaises(FileNotFoundError):
            resolve_robot_data_path("missing_robot_dataset")


if __name__ == "__main__":
    unittest.main()
