import argparse
import sys
import os
import subprocess

root = os.getcwd()
datapath = os.path.join(root,'./data')

parser = argparse.ArgumentParser(description="根据参数执行不同功能")
parser.add_argument('--mode', '-m', required=True, 
                choices=['pretreat', 'stat_select', 'evalate'],
                help='执行内容')
parser.add_argument('--dimension', '-d', help='选择的维度')
args = parser.parse_args()

def main():
    if args.mode == 'pretreat':
        subprocess.run(["python", "./src/pretreat.py", datapath])
    elif args.mode == 'stat_select':
        if args.dimension:
            subprocess.run(["python", "stat_select.py"])
        else:
            print('-d is needed')
        
    elif args.mode == 'evalate':
        subprocess.run(["python", "evalate.py"])
    else:
        print(f"未知模式: {args.mode}")

if __name__ == "__main__":
    main()