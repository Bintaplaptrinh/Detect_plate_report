import argparse

from knn_model import load_training_data, save_knn, train_knn


def main() -> int:
    parser = argparse.ArgumentParser(description="Train KNN from txt files and save an OpenCV model.")
    parser.add_argument("--classifications", default="classifications.txt")
    parser.add_argument("--flattened", default="flattened_images.txt")
    parser.add_argument("--out", default="knn_model.xml")
    args = parser.parse_args()

    flattened, classifications = load_training_data(
        classifications_path=args.classifications,
        flattened_images_path=args.flattened,
    )

    knn = train_knn(flattened, classifications)
    save_knn(knn, args.out)
    print(f"Saved KNN model to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
