"""06_validation_errors.py — Typed validation errors in linoss_dynamics.

Each error subclass targets a specific class of invalid input so callers can
catch fine-grained exceptions.  This script triggers each one intentionally and
catches it to show the error message.
"""

from __future__ import annotations

import numpy as np

from linoss_dynamics import (
    InvalidDampingError,
    InvalidShapeError,
    LinOSSError,
    UnsupportedModeError,
    linoss_step,
)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    print("=== 06 Validation Errors ===")
    print()
    print("Error hierarchy:")
    print("  LinOSSError (base)")
    print("    ├── InvalidShapeError")
    print("    ├── InvalidDampingError")
    print("    └── UnsupportedModeError")

    # 1. InvalidShapeError — y and z have mismatched lengths
    section("InvalidShapeError: mismatched y and z")
    y_bad = np.array([1.0, 2.0])       # length 2
    z_bad = np.array([0.0, 0.0, 0.0])  # length 3 — mismatch!
    A = np.array([1.0, 4.0, 9.0])
    try:
        linoss_step(y_bad, z_bad, A, dt=0.1)
    except InvalidShapeError as exc:
        print(f"Caught InvalidShapeError: {exc}")
    except LinOSSError as exc:
        print(f"Caught LinOSSError (unexpected subtype): {exc}")

    # 2. InvalidDampingError — negative damping coefficient G
    section("InvalidDampingError: negative G")
    y_ok = np.array([1.0])
    z_ok = np.array([0.0])
    A_ok = np.array([1.0])
    G_neg = np.array([-0.5])   # negative damping is physically meaningless
    try:
        linoss_step(y_ok, z_ok, A_ok, dt=0.1, damping=G_neg)
    except InvalidDampingError as exc:
        print(f"Caught InvalidDampingError: {exc}")
    except LinOSSError as exc:
        print(f"Caught LinOSSError (unexpected subtype): {exc}")

    # 3. UnsupportedModeError — unrecognised discretization scheme
    section("UnsupportedModeError: unknown mode string")
    try:
        linoss_step(y_ok, z_ok, A_ok, dt=0.1, mode="euler")
    except UnsupportedModeError as exc:
        print(f"Caught UnsupportedModeError: {exc}")
    except LinOSSError as exc:
        print(f"Caught LinOSSError (unexpected subtype): {exc}")

    # 4. All three are subclasses of LinOSSError — catch them with a single clause.
    section("Catching all errors with the base LinOSSError")
    errors_to_try = [
        (y_bad, z_bad, A, 0.1, "implicit", None),
        (y_ok, z_ok, A_ok, 0.1, "implicit", G_neg),
        (y_ok, z_ok, A_ok, 0.1, "bogus_mode", None),
    ]
    for y_, z_, A_, dt_, mode_, G_ in errors_to_try:
        try:
            linoss_step(y_, z_, A_, dt=dt_, mode=mode_, damping=G_)
        except LinOSSError as exc:
            print(f"  {type(exc).__name__}: {exc}")

    print("\nAll validation errors handled cleanly.")


if __name__ == "__main__":
    main()
