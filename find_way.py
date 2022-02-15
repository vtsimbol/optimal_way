import argparse

from core import WayFinder, init_logger

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True, help='The path to the warehouse annotation')
args = parser.parse_args()

init_logger()


if __name__ == '__main__':
    finder = WayFinder(anno_path=args.input_file)
    finder()
