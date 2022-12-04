#!/bin/bash
# hack to avoid home dir problem
export HOME=${_CONDOR_SCRATCH_DIR}
#export
/usr/bin/python3 /application/run.py "$@"