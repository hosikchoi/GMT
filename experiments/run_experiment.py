import os
import numpy as np
import pandas as pd
import argparse

def main(method: str, n_trials: int = 20):
    scores = []

    print(f"  Running {n_trials} experiments for method: {method}")

    for i in range(n_trials):
        # 실행
        result = os.popen(f"python -m experiments.train_{method} --random_state_value={i}").read()

        # R² 추출
        try:
            score_line = [line for line in result.split('\n') if "R² score" in line][0]
            score = float(score_line.split(":")[1])
            scores.append({"seed": i, "r2": score})
        except Exception as e:
            print(f"[{i}] Error parsing: {e}")
            continue

        # 10회마다 로그 출력
        if i % 10 == 0:
            print(f"[{i}] R² score: {score:.4f}")

    # 결과 저장
    os.makedirs("results", exist_ok=True)
    filename = f"results/results_{method}.csv"

    df = pd.DataFrame(scores)
    df.to_csv(filename, index=False)

    # 요약 통계
    print(f"\n {len(scores)} runs completed.")
    print(f"Results saved to: {filename}")
    print(f"Mean R²: {df['r2'].mean():.4f}, Std: {df['r2'].std():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        help="Method name: fone, gmt, macode 등")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of repeated runs (default: 20)")
    args = parser.parse_args()

    main(method=args.method, n_trials=args.n_trials)

#print("Step 3: Run MaCoDE baseline")
#os.system("python -m experiments.train_macode")

#print("Step 4: Run FoNE baseline")
#os.system("python -m experiments.train_fone")

##os.system("python models/train_fone.py")
