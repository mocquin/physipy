This site contains the project documentation for the
[`physipy`](https://github.com/mocquin/physipy/) project.

## Playground

```py play
from physipy import units, constants

meter = units['m']
speed_of_light = constants['c']

length = np.arange(10) * m
travel_time = length / speed_of_light
print(travel_time)
```

{!./../README.md!}
