"""Convert SMPL pkl files to remove chumpy dependency.

Run once: python convert_smpl.py
"""
import copyreg
import pickle
import numpy as np
import sys
import types
import os

# ------------------------------------------------------------------
# Monkey-patch copyreg._reconstructor to handle ndarray subclasses
# ------------------------------------------------------------------
_orig_reconstructor = copyreg._reconstructor


def _patched_reconstructor(cls, base, state):
    if issubclass(cls, np.ndarray):
        obj = np.ndarray.__new__(cls, shape=(0,))
        # Don't call __setstate__ here; let pickle do it with proper logic
        return obj
    return _orig_reconstructor(cls, base, state)


copyreg._reconstructor = _patched_reconstructor


# ------------------------------------------------------------------
# A subclass that can absorb chumpy's dict state
# ------------------------------------------------------------------
class _ChStub(np.ndarray):
    """Absorbs chumpy array pickle state and converts to ndarray."""

    def __new__(cls, *args, **kwargs):
        return np.ndarray.__new__(cls, shape=(0,))

    def __setstate__(self, state):
        # chumpy objects pickle their state as a dict; numpy uses a tuple
        if isinstance(state, dict):
            # The actual array data is often in 'x' or just return empty
            if 'x' in state:
                arr = np.asarray(state['x'])
                # Resize self to match
                super().__setstate__((1, arr.shape, arr.dtype, False, arr.tobytes()))
            # else: leave as empty array
        elif isinstance(state, tuple):
            try:
                super().__setstate__(state)
            except Exception:
                pass

    def __reduce__(self):
        return (np.array, (np.asarray(self),))

    def __reduce_ex__(self, protocol):
        return self.__reduce__()


# ------------------------------------------------------------------
# Stub out chumpy
# ------------------------------------------------------------------
class _FakeModule(types.ModuleType):
    def __getattr__(self, name):
        return _ChStub


for mod_name in [
    'chumpy', 'chumpy.ch', 'chumpy.ch_ops', 'chumpy.utils',
    'chumpy.logic', 'chumpy.reordering',
]:
    sys.modules[mod_name] = _FakeModule(mod_name)


class _NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if 'chumpy' in module:
            return _ChStub
        return super().find_class(module, name)


def convert_file(src: str, dst: str) -> None:
    print(f"  Loading {src} ...")
    with open(src, "rb") as f:
        data = _NumpyUnpickler(f, encoding="latin1").load()

    clean = {}
    for k, v in data.items():
        if hasattr(v, "toarray"):  # scipy sparse
            clean[k] = v
        elif isinstance(v, np.ndarray):
            clean[k] = np.asarray(v)
        elif hasattr(v, "__array__"):
            clean[k] = np.asarray(v)
        else:
            clean[k] = v

    print(f"  Keys: {list(clean.keys())}")
    for k, v in clean.items():
        if isinstance(v, np.ndarray):
            print(f"    {k}: ndarray {v.shape} {v.dtype}")
        else:
            print(f"    {k}: {type(v).__name__}")

    print(f"  Saving {dst} ({len(clean)} keys) ...")
    with open(dst, "wb") as f:
        pickle.dump(clean, f, protocol=2)

    # Verify
    copyreg._reconstructor = _orig_reconstructor
    with open(dst, "rb") as f:
        test = pickle.load(f)
    copyreg._reconstructor = _patched_reconstructor
    print(f"  Verified: {len(test)} keys load OK")


def main():
    smpl_dir = os.path.join("app", "assets", "smpl")
    for fname in ["SMPL_MALE.pkl", "SMPL_FEMALE.pkl", "SMPL_NEUTRAL.pkl"]:
        fpath = os.path.join(smpl_dir, fname)
        if not os.path.exists(fpath):
            print(f"  Skipping {fname} (not found)")
            continue
        convert_file(fpath, fpath)
    print("All done.")


if __name__ == "__main__":
    main()
