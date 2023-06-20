During this project, the aim was to pickle models and input data for use in subsequent ipynb files.

Unfortunately, this approach is not directly compatible with one of the key modeling processes used: the multi-estimator pipeline, which includes custom functions for vector preprocessing.

These files are retained, in addition to the code which generates pickles on model fit, to enable conversion to .py files at a later stage.