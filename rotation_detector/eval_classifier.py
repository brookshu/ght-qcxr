import pandas as pd
import argparse

# import your inference helper
from inference.rotation_infer import load_rotation_model, predict_rotation


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="labels CSV (path,label)")
    parser.add_argument("--model_dir", required=True, help="trained model folder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args()

    print("\nLoading rotation model...")
    rot_model = load_rotation_model(args.model_dir, device=args.device)

    df = pd.read_csv(args.csv)

    total = 0
    correct = 0

    print("\nRunning evaluation...\n")

    for _, row in df.iterrows():

        #path = row["path"]
        path = "training/" + row["path"]
        gt_label = int(row["label"])

        result = predict_rotation(
            rot_model,
            path,
            threshold=args.threshold
        )

        prob = result["p_rotated"]
        pred = int(result["rotation_detected"])

        is_correct = (pred == gt_label)

        if is_correct:
            correct += 1

        total += 1

        print(
            f"{path} | "
            f"GT={gt_label} | "
            f"Pred={pred} | "
            f"Prob={prob:.3f} | "
            f"{'✅' if is_correct else '❌'}"
        )

    print("\n=============================")
    print(f"Accuracy: {correct}/{total} = {correct/total:.3f}")
    print("=============================\n")


if __name__ == "__main__":
    main()