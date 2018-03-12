import argparse, csv, math, sys
from collections import Counter, namedtuple
import numpy as np

Position = namedtuple('Position', ['x', 'y'])
Snapshot = namedtuple('Snapshot', ['team_1', 'team_2'])

# the basket is located 1.60m in front of the base line
LEFT_BASKET = Position(x=-12.75, y=0)


def euclidean_distance(a, b):
    return math.sqrt((a.x - b.x) ** 2.0 + (a.y - b.y) ** 2.0)


def average_distance_to_position(positions, position):
    return np.average([euclidean_distance(a, position) for a in positions])


def distance_to_left_basket(team_positions):
    return average_distance_to_position(team_positions, LEFT_BASKET)


def log_and_remove_incomplete_snapshots(result):
    incomplete_measurements = []
    for k, snapshot in result.items():
        if not snapshot.team_1:
            sys.stderr.write(f'no data for team 1 at {k} ms\n')
            incomplete_measurements.append(k)
        if not snapshot.team_2:
            sys.stderr.write(f'no data for team 2 at {k} ms\n')
            incomplete_measurements.append(k)

    for i in incomplete_measurements:
        del result[i]


def read_snippet(file_path):
    result = {}

    with open(file_path, 'rt') as f:
        for row in csv.DictReader(f):
            key = int(row['ts in ms'])
            team = int(row['group id'])

            position = Position(x=float(row['x in m']), y=float(row['y in m']))
            snapshot = Snapshot(team_1=[], team_2=[])

            if key in result:
                snapshot = result[key]
            else:
                result[key] = snapshot

            if team == 1:
                snapshot.team_1.append(position)
            else:
                snapshot.team_2.append(position)

    log_and_remove_incomplete_snapshots(result)

    return list(result.values())


def team_currently_closer_to_left_basket(snapshot):
    if distance_to_left_basket(snapshot.team_1) < distance_to_left_basket(snapshot.team_2):
        return 1
    else:
        return 2


def team_defending_left_basket(snippet):
    count = Counter(map(team_currently_closer_to_left_basket, snippet))
    return count.most_common()[0][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect the baseline of team 1 in a basketball game.')
    parser.add_argument('filepath', help='path to a csv formatted file of basketball player position data')
    args = parser.parse_args()
    snippet = read_snippet(args.filepath)

    side = 'left' if team_defending_left_basket(snippet) == 1 else 'right'
    print(f'The team with group id 1 has its baseline on the {side} side')