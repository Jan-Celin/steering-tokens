import argparse
import sys
import train
import eval

def main():
    parser = argparse.ArgumentParser(description="Steering Tokens Pipeline")
    parser.add_argument("mode", choices=["train", "eval"], help="Run training or evaluation")
    
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    
    if args.mode == "train":
        train.main()
    elif args.mode == "eval":
        eval.main()

if __name__ == "__main__":
    main()
