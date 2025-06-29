import subprocess
from itertools import product


def main():
    models = ["swin_v2_s", "effnet_b7", "densenet201", "googlenet"]
    with_synthetic = [True, False]
    with_aug = [True, False]
    with_clahe = [True, False]

    for model, with_synthetic, with_aug, with_clahe in product(
        models, with_synthetic, with_aug, with_clahe
    ):
        cmd = [
            "python",
            "scripts/train_depth_regression.py",
            "--model_name",
            model,
        ]

        if with_synthetic:
            cmd.append("--with_synthetic")
        if with_aug:
            cmd.append("--with_aug")
        if with_clahe:
            cmd.append("--with_clahe")

        print(" ".join(cmd))

        subprocess.run(cmd)


if __name__ == "__main__":
    main()
