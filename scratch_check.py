import digital_rf as drf
import h5py

ROOT = r"C:\Users\hatti\Downloads\netbeans-26-bin\sds-code\sdk\tests\integration\data\captures\drf\westford-vpol"
r = drf.DigitalRFReader(ROOT)

print("Channels:", r.get_channels())

for ch in r.get_channels():
    start, end = r.get_bounds(ch)
    print(ch, "bounds:", start, "-", end, "total samples:", end - start, ")")
    x = r.read_vector(start, min(100000, end-start), ch)
    print("read:", x.shape, "dtype:", x.dtype)

path = r"C:\Users\hatti\Downloads\netbeans-26-bin\sds-code\sdk\tests\integration\data\captures\drf\westford-vpol\cap-2024-06-27T14-00-00\metadata\2024-06-27T14-00-00\metadata@1719499588.h5"
with h5py.File(path, 'r') as f:
    print("FILE ATTRIBUTES:")
    for k, v in f.attrs.items():
        print(k, ":", v)

    print("DATASETS:")
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"\n{name}:")
            print(obj[()])
    f.visititems(walk)
