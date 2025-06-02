import h5py

with h5py.File("vlad_vectors.hdf5", "r") as f:
    print("Datasets:", list(f.keys()))
    if "vectors" in f:
        print("Shape:", f["vectors"].shape)
        print("First row:", f["vectors"][0])


