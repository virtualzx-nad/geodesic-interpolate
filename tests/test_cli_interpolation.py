import subprocess
import os
import pytest


@pytest.fixture(autouse=True)
def cleanup_output():
    output_file = "interpolated.xyz"
    # Remove before test
    if os.path.exists(output_file):
        os.remove(output_file)
    yield
    # Remove after test
    if os.path.exists(output_file):
        os.remove(output_file)


def test_interpolated_xyz_geometry_count():
    result1 = subprocess.run(
        [
            "geodesic_interpolate",
            "--nimages", "5",
            "--seed", "123",
            "test_cases/DielsAlder.xyz"
        ],
        capture_output=True,
        text=True
    )
    result2 = subprocess.run(
        [
            "geodesic_interpolate",
            "--nimages", "5",
            "--seed", "123",
            "test_cases/DielsAlder.xyz"
        ],
        capture_output=True,
        text=True
    )
    result3 = subprocess.run(
        [
            "geodesic_interpolate",
            "--nimages", "5",
            "--seed", "1234",
            "test_cases/DielsAlder.xyz"
        ],
        capture_output=True,
        text=True
    )
    output1 = result1.stdout + result1.stderr
    output2 = result2.stdout + result2.stderr
    output3 = result3.stdout + result3.stderr
    
    # Assert that same seed gives same output
    assert output1 == output2, "Outputs with the same seed should match"

    # Assert that different seed gives different output
    assert output1 != output3

    # Ensure all subprocess calls completed successfully (exit code 0 means no errors)
    assert result1.returncode == result2.returncode == result3.returncode == 0

    assert f"Seed: 123" in output1
    assert f"Seed: 123" in output2
    assert f"Seed: 1234" in output3
