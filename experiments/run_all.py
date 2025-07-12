import os

def run(title, command):
    print(f"\n\033[94m[RUNNING] {title}\033[0m")
    code = os.system(command)
    if code != 0:
        print(f"\033[91m[ERROR] Failed: {command}\033[0m")
        exit(1)

if __name__ == "__main__":
    print("\n Starting full experiment pipeline...\n")

    # Step 1: Data generation
    run("Step 1: Generate synthetic data", "python data/generate_data.py")

    # Step 2: GMT
    run("Step 2A: Pretrain GMT-BERT", "python models/pretrain_gmt.py")
    run("Step 2B: Downstream regression with GMT", "python models/downstream.py")

    # Step 3: FoNE
    run("Step 3A: FoNE direct embedding (no BERT)", "python models/train_fone.py")  # FoNE 방식은 pretrain 필요 없음

    # Step 4: MaCoDE
    run("Step 4A: Pretrain MaCoDE", "python models/pretrain_macode.py")
    run("Step 4B: Downstream regression with MaCoDE", "python models/downstream_macode.py")

    print("\n✅ All experiments completed successfully!")
