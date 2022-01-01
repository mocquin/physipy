
"""
To run locally : 
1. Install required packages : 
use github to get version 0.5 that has not been published to pypi:
pip install git+git://github.com/airspeed-velocity/asv@master
```
pip install virtualenv
```
2. Once setup and benchmarks written : (this may take a while)
```
asv run ALL  # to run all benchmarks on all comiits
```
3. Convert to html and inspect
```
asv publish  # convert results to html
asv preview  # display html on server
```

"""