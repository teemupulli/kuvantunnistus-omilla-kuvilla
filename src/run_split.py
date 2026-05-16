import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from data_pipeline import split_dataset

if __name__ == '__main__':
    src_dir = Path('..') / 'dataset'
    src_dir = src_dir.resolve()
    target_dir = Path('..') / 'dataset' / 'split'
    target_dir = target_dir.resolve()

    out = split_dataset(str(src_dir), str(target_dir), train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=123)
    print('Dataset split completed into:', out)

    # Print counts
    for subset in ['train', 'val', 'test']:
        subset_dir = target_dir / subset
        if subset_dir.exists():
            print(f"\nSubset: {subset}")
            for cls in sorted([d for d in subset_dir.iterdir() if d.is_dir()]):
                n = len(list(cls.iterdir()))
                print(f"  {cls.name}: {n} images")
        else:
            print(f"Subset {subset} not found")
