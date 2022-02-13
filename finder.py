import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True, help='The path to the warehouse annotation')
args = parser.parse_args()


class WayFinder:
    def __init__(self, anno_path: str):
        pass

    def __call__(self):
        pass


if __name__ == '__main__':
    finder = WayFinder(anno_path=args.input_file)
    finder()
