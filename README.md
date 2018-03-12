# Basketball Team Direction

## Outline

Small helper script to detect the playing direction of team 1 in a basketball game based upon location sensor data from the players only (i.e. location of the ball is unknown).

The general idea is that teams "on average" stand closer to their own basket than their opponents basket "most of the time".

To detect if this is the case during each 'snapshot' (i.e. the timestamp) the euclidean distance of each player to the 'left basket' (not baseline!) is calculated. The distances are averaged for both teams.

The team with the lower average distance to the left basket is considered to be closer to the basket. Each snapshot where this can be observed is treated as *evidence* for the 'nearer' team being the team defending the left basket and as such having the left side as their base line.

The team which most often happens to be nearer the left basket during the 60 seconds of one snipped is considered to be the team that has it's baseline on the left side.

The approach – even though naive – should be robust given assuming that it is fed a long enough time frame of a play and that both teams behave "rational" (i.e. should the defensive team usually rush into the backcourt, this approach would assign both teams to the wrong baseline).

It is not robust for short time snippets of special phases in the game, e.g. a player of the attacking team blocking on the inside of a defending player to prevent the defending player from interfering with another attacking player's rush to the basket/restricted zone. During this time you could have two of the 5 players of the attacking team be close enough to the defending teams basket that the team's average is below that of the defenders.

## Improvements

### Team Formation

It is reasonable to assume that the defending team tries to keep a closer formation than the attacking team while the attacking team tries to spread out in order to open enough space for a break to the opponents basket.

Tightness could e.g. calculated via the average distances between each two players of a single team:

```Python
import itertools as it
import numpy as np

def average_distance_between_positions(positions):
    return np.average([euclidean_distance(a, b) for a, b in it.combinations(positions, 2)])
```

Knowing which team is in a tighter formation could be used as additional evidence for a naive bayesian filter (i.e. if both teams are closer to the left basket and team 1 is closer to the left basket than team 2 and team 1 has a tighter formation it is more likely that team 1 defends the left basket, than if team 1 had a more spread formation).

Implementing a naive bayesian classifier would rapidly approach a very high probability for one side even when given very small differences in probability between the observations (i.e. 50,1% vs. 49,9% for the closer side being the baseline of the observed team).

```Python
import functools

def bayesian_classifier(observations):
    if len(observations) == 1:
        return observations[0]

    def reducer(acc, n):
        normalizing_constant = 0.0
        for k, v in n.items():
            normalizing_constant += v * acc[k]

        def mapper(i):
            k, prior = i
            likelihood = acc[k]
            posterior = prior * likelihood / normalizing_constant
            return (k, posterior)

        return {k:v for k,v in map(mapper, n.items())}

    return functools.reduce(reducer, observations)

# probabilities of 0.501 and 0.499 for distance have been arbitrarily set
observations_distance = [{'left': 0.501, 'right': 0.499}] * 120
# probabilities of 0.501 and 0.499 for tightness have been arbitrarily set
observations_tightness = [{'left': 0.505, 'right': 0.495}] * 120
result = bayesian_classifier(observations_distance + observations_tightness)
# -> {'left': 0.946852921993493, 'right': 0.05314707800650693}
```

The bayesian approach would allow for additional evidence to be integrated, e.g. the defending team is usually mimicking the attacking teams movements and not vice versa.

## Execution

The script assumes that Python 3.6 and Numpy 1.13.3 are installed on the system (`environment.yml` provides the description for a `conda` environment)

### Setup

To install the required dependencies do the following in the `solution` directory:

```bash
> conda update conda
> source activate basketball-env
```

To ensure that numpy is installed run:

```bash
> conda list | grep numpy
```

### Running

The script can be executed via:

```bash
> python main.py sensor_data/snippet1.csv
```

## Results

```bash
> python main.py sensor_data/snippet1.csv
The team with group id 1 has its baseline on the left side

> python main.py sensor_data/snippet2.csv
The team with group id 1 has its baseline on the right side

> python main.py sensor_data/snippet3.csv
The team with group id 1 has its baseline on the right side

> python main.py sensor_data/snippet4.csv
The team with group id 1 has its baseline on the right side

> python main.py sensor_data/snippet5.csv
The team with group id 1 has its baseline on the left side
```

## Remarks

Snippets 2 and 3 produce warnings, because there are no measurements for team 1 or 2 available at some time stamps (the script logs the timestamp to `stderr` and then removes the observations before proceeding). The missing data should not influence the results.

The solution is not robust against common input errors, the only issue I handled was the one I ran in with the provided data (i.e. it's not production code, but I assume that was not part of the challenge).

I also choose to use 3rd party library functions for uninteresting parts of the challenge and implementing the interesting parts in plain Python myself (instead of using working e.g. with Pandas data frames).