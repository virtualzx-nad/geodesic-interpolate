import contextlib
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from geodesic_interpolate import __main__ as cli
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

    def test_optimizer_errors_and_interrupts_do_not_write_final_output(self):
        for sweep in [False, True]:
            for error in [ValueError("failed optimization"), KeyboardInterrupt()]:
                with self.subTest(sweep=sweep, error=type(error).__name__):
                    self.output.write_text("preserve final output")
                    kwargs = {"sweep_effect" if sweep else "smooth_effect": error}
                    with self.assertRaises(type(error)):
                        self.run_cli(["--sweep"] if sweep else [], **kwargs)
                    self.assertEqual(self.output.read_text(), "preserve final output")

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

    def test_valid_raw_path_is_written_on_request(self):
        _, _, path = self.run_cli(["--save-raw", str(self.raw_output)])
        _, raw = read_xyz(self.raw_output)
        np.testing.assert_array_equal(raw, path)


if __name__ == "__main__":
    unittest.main()
