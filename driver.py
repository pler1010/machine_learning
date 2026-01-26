import argparse
import sys
import os
import subprocess

root = os.getcwd()
datapath = os.path.join(root,'./data')

parser = argparse.ArgumentParser(description="根据参数执行不同功能")
parser.add_argument('--mode', '-m', required=True, 
                choices=['pretreat', 'stat_select',  'ml_select', 'evaluate'],
                help='执行内容')
parser.add_argument('--dimension', '-d', help='选择的维度')
parser.add_argument('--alpha', '-a', help='RF 与 XGB 融合权重 alpha (0~1)', default="0.5")
args = parser.parse_args()

def main():
    if args.mode == 'pretreat':
        subprocess.run(["python", "./src/pretreat.py", datapath])
    elif args.mode == 'stat_select':
        if args.dimension:
            subprocess.run(["python", "./src/stat_select.py", datapath, args.dimension])
        else:
            print('-d is needed')

    elif args.mode == 'ml_select':
        # 默认 Top100
        topn = args.dimension if args.dimension else "100"
        subprocess.run(["python", "./src/ml_select.py", datapath, topn, args.alpha])

        
    elif args.mode == 'evaluate':
        subprocess.run(["python", "./src/evaluate.py", datapath])
    else:
        print(f"未知模式: {args.mode}")

if __name__ == "__main__":
    main()