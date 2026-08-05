"""Pytest config / runner for equity-conviction-monitor parity gates.

Mirrors crypto-conviction-monitor/tests/test_parity.py runner:
runs the three checks and prints a clear banner, so CI + manual runs see
exactly which gate failed (parity / perp-overlay / frozen-regression).
"""
import sys, importlib

def check_list():
    mod = importlib.import_module("tests.test_parity")
    return [
        ("frontend/backend parity", mod.check_frontend_backend_parity),
        ("frozen conviction regression", mod.check_frozen_conviction),
    ]

def run() -> bool:
    ok = True
    for name, fn in check_list():
        try:
            fn(); print(f"  PASS  {name}")
        except Exception as e:
            ok = False; print(f"  FAIL  {name}: {e}")
    print("\nPARITY:", "ALL PASS" if ok else "FAILURES")
    return ok

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
