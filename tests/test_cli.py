import contextlib
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from geodesic_interpolate import __main__ as cli
from geodesic_interpolate import geodesic as geodesic_module
from geodesic_interpolate.fileio import read_xyz
from geodesic_interpolate.validation import UnsafePathError


class CLITest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.output = pathlib.Path(self.tempdir.name) / "output.xyz"
        self.raw_output = pathlib.Path(self.tempdir.name) / "raw.xyz"

    def run_cli(self, options=(), natoms=2, path=None, smooth_effect=None,
                sweep_effect=None):
        atoms = ["C"] * natoms
        if path is None:
            geometry = np.zeros((natoms, 3))
            geometry[:, 0] = np.arange(natoms) * 4.
            path = np.repeat(geometry[None], 3, axis=0)
        smoother = mock.Mock(path=path.copy(), rij_list=[])
        smoother.smooth.side_effect = smooth_effect
        smoother.sweep.side_effect = sweep_effect
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch("sys.argv", [
                "geodesic_interpolate", "input.xyz", "--output", str(self.output), *options]))
            stack.enter_context(mock.patch.object(cli, "read_xyz", return_value=(atoms, path)))
            stack.enter_context(mock.patch.object(cli, "redistribute", return_value=path))
            constructor = stack.enter_context(mock.patch.object(cli, "Geodesic", return_value=smoother))
            cli.main()
        return smoother, constructor, path

    def test_default_global_smoothing_and_50_iterations_for_any_atom_count(self):
        for natoms in [2, 36, 100]:
            with self.subTest(natoms=natoms):
                smoother, constructor, path = self.run_cli(natoms=natoms)
                smoother.smooth.assert_called_once_with(tol=2e-3, max_iter=50)
                smoother.sweep.assert_not_called()
                self.assertIs(constructor.call_args.kwargs["align"], False)
                _, written = read_xyz(self.output)
                np.testing.assert_array_equal(written, path)

    def test_explicit_sweep_and_no_sweep_are_retained(self):
        smoother, _, _ = self.run_cli(["--sweep", "--maxiter", "9", "--microiter", "3"])
        smoother.sweep.assert_called_once_with(tol=2e-3, max_iter=9, micro_iter=3)
        smoother.smooth.assert_not_called()
        smoother, _, _ = self.run_cli(["--no-sweep", "--maxiter", "8"], natoms=40)
        smoother.smooth.assert_called_once_with(tol=2e-3, max_iter=8)
        smoother.sweep.assert_not_called()

    def test_invalid_initial_midpoint_never_creates_or_overwrites_outputs(self):
        path = np.array([[[0., 0., 0.], [2., 0., 0.]],
                         [[2., 0., 0.], [0., 0., 0.]]])
        for existing in [False, True]:
            with self.subTest(existing=existing):
                if existing:
                    self.output.write_text("preserve final output")
                    self.raw_output.write_text("preserve raw output")
                with self.assertRaisesRegex(UnsafePathError, "midpoint"):
                    self.run_cli(["--save-raw", str(self.raw_output)], path=path)
                if existing:
                    self.assertEqual(self.output.read_text(), "preserve final output")
                    self.assertEqual(self.raw_output.read_text(), "preserve raw output")
                else:
                    self.assertFalse(self.output.exists())
                    self.assertFalse(self.raw_output.exists())

    def test_optimizer_errors_do_not_write_final_output(self):
        for sweep in [False, True]:
            with self.subTest(sweep=sweep):
                self.output.write_text("preserve final output")
                kwargs = {"sweep_effect" if sweep else "smooth_effect": ValueError("failed optimization")}
                with self.assertRaisesRegex(ValueError, "failed optimization"):
                    self.run_cli(["--sweep"] if sweep else [], **kwargs)
                self.assertEqual(self.output.read_text(), "preserve final output")

    def test_interrupt_saves_validated_path_and_is_reraised(self):
        path = np.array([[[0., 0., 0.], [2., 0., 0.]]] * 3)
        for sweep in [False, True]:
            with self.subTest(sweep=sweep):
                kwargs = {"sweep_effect" if sweep else "smooth_effect": KeyboardInterrupt()}
                with self.assertRaises(KeyboardInterrupt):
                    self.run_cli(["--sweep"] if sweep else [], path=path, **kwargs)
                _, written = read_xyz(self.output)
                np.testing.assert_array_equal(written, path)

    def test_interrupted_sweep_saves_completed_work_and_restores_active_trial(self):
        # Exercise the actual CLI, redistribution, sweep, and local smooth
        # rollback. Only inject interruption into a second-sweep solver trial.
        filename = pathlib.Path(__file__).resolve().parents[1] / "test_cases" / "DielsAlder_interpolated.xyz"
        original_redistribute = cli.redistribute
        original_smooth = geodesic_module.Geodesic.smooth
        snapshots = {}
        visited = []
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)

        def redistribute(*args, **kwargs):
            raw = original_redistribute(*args, **kwargs)
            snapshots["prepared"] = np.array(raw, copy=True)
            return raw

        def smooth(smoother, *args, **kwargs):
            visited.append(kwargs["start"])
            if len(visited) > 8:
                def interrupt_solver(fun, x0, jac, **options):
                    trial = x0.reshape(-1, 3).copy()
                    trial[:, 1] += .001
                    fun(trial.ravel(), **options["kwargs"])
                    self.assertFalse(np.array_equal(smoother.path, snapshots["completed"]))
                    raise KeyboardInterrupt("injected during the second sweep")

                with mock.patch.object(geodesic_module, "least_squares", side_effect=interrupt_solver):
                    return original_smooth(smoother, *args, **kwargs)
            result = original_smooth(smoother, *args, **kwargs)
            if len(visited) == 8:
                snapshots["completed"] = smoother.path.copy()
                snapshots["smoother"] = smoother
            return result

        with mock.patch("sys.argv", [
                "geodesic_interpolate", str(filename), "--output", str(self.output),
                "--nimages", "10", "--sweep", "--microiter", "5", "--maxiter", "50",
                "--tol", "0.000001"]), \
                mock.patch.object(cli, "redistribute", side_effect=redistribute), \
                mock.patch.object(geodesic_module.Geodesic, "smooth", new=smooth):
            with self.assertRaisesRegex(KeyboardInterrupt, "second sweep"):
                cli.main()
        self.assertEqual(visited[:8], list(range(1, 9)))
        self.assertGreater(len(visited), 8)
        self.assertFalse(np.array_equal(snapshots["completed"], snapshots["prepared"]))
        np.testing.assert_array_equal(snapshots["smoother"].path, snapshots["completed"])
        np.testing.assert_array_equal(snapshots["smoother"].path[[0, -1]], snapshots["prepared"][[0, -1]])
        _, written = read_xyz(self.output)
        # XYZ output rounds each Cartesian coordinate to twelve decimal places.
        np.testing.assert_allclose(written, snapshots["completed"], rtol=0., atol=5.01e-13)

    def test_invalid_final_path_does_not_overwrite_output(self):
        atoms = ["C", "C"]
        path = np.array([[[0., 0., 0.], [2., 0., 0.]]] * 3)
        smoother = mock.Mock(path=path.copy(), rij_list=[])

        def produce_overlap(**kwargs):
            smoother.path[1, 1, 0] = .1

        smoother.smooth.side_effect = produce_overlap
        self.output.write_text("preserve final output")
        with mock.patch("sys.argv", ["geodesic_interpolate", "input.xyz", "--output", str(self.output)]), \
                mock.patch.object(cli, "read_xyz", return_value=(atoms, path)), \
                mock.patch.object(cli, "redistribute", return_value=path), \
                mock.patch.object(cli, "Geodesic", return_value=smoother):
            with self.assertRaisesRegex(UnsafePathError, "image 1"):
                cli.main()
        self.assertEqual(self.output.read_text(), "preserve final output")

    def test_unsafe_interrupted_path_does_not_overwrite_output(self):
        atoms = ["C", "C"]
        path = np.array([[[0., 0., 0.], [2., 0., 0.]]] * 3)
        for sweep in [False, True]:
            with self.subTest(sweep=sweep):
                smoother = mock.Mock(path=path.copy(), rij_list=[])

                def unsafe_interrupt(**kwargs):
                    smoother.path[1, 1, 0] = .1
                    raise KeyboardInterrupt()

                method = smoother.sweep if sweep else smoother.smooth
                method.side_effect = unsafe_interrupt
                self.output.write_text("preserve final output")
                with mock.patch("sys.argv", [
                        "geodesic_interpolate", "input.xyz", "--output", str(self.output),
                        *(["--sweep"] if sweep else [])]), \
                        mock.patch.object(cli, "read_xyz", return_value=(atoms, path)), \
                        mock.patch.object(cli, "redistribute", return_value=path), \
                        mock.patch.object(cli, "Geodesic", return_value=smoother):
                    with self.assertRaisesRegex(UnsafePathError, "image 1"):
                        cli.main()
                self.assertEqual(self.output.read_text(), "preserve final output")

    def test_valid_raw_path_is_written_on_request(self):
        _, _, path = self.run_cli(["--save-raw", str(self.raw_output)])
        _, raw = read_xyz(self.raw_output)
        np.testing.assert_array_equal(raw, path)


if __name__ == "__main__":
    unittest.main()
