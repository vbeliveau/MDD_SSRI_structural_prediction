#!/usr/bin/env python3

import os
import argparse
import inspect
import shlex

import numpy as np

from subprocess import Popen, PIPE
from os.path import join, abspath, isfile, isdir
from multiprocessing import Pool

# Default paths
main_dir = os.path.dirname(abspath(join(inspect.getfile(inspect.currentframe()))))


def run(cmd, live_verbose=False):
    print('\n' + cmd)
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    output, error = p.communicate()
    if output:
        print(output.decode('latin_1'))
    if error:
        print(error.decode('latin_1'))


def fastsurfer_segmentation(subject, recon_dir, t1_dir):

    """ Run FastSurfer segmentation inside docker container """

    seg_file = join(recon_dir, subject, 'mri', 'aparc.DKTatlas+aseg.deep.mgz')

    if not isfile(seg_file):
        run('docker run -it --rm \
        	--gpus all \
            -v ' + abspath(main_dir) + ':/data \
            -v ' + abspath(recon_dir) + ':/data/recon-all \
            -v ' + abspath(t1_dir) + ':/data/t1 \
           	fastsurfer:gpu \
        	--fs_license /data/license \
        	--t1 /data/t1/' + subject + '.nii.gz \
        	--sid ' + subject + \
        	' --sd /data/recon-all \
            --seg_only'
        )
    else:
        print('File ' + seg_file + ' already exists. Skipping.')


def fastsurfer_surface(subject, recon_dir, t1_dir, cpus=1):

    """ Run FastSurfer surface recon inside docker container """

    stat_file = join(recon_dir, subject, 'stats', 'rh.aparc.stats')

    if not isfile(stat_file):
        run(
            'docker run -it --rm \
            -v ' + abspath(main_dir) + ':/data \
            -v ' + abspath(recon_dir) + ':/data/recon-all \
            -v ' + abspath(t1_dir) + ':/data/t1 \
            --cpus=' + str(cpus) + \
           	' fastsurfer:cpu \
        	--fs_license /data/license \
        	--t1 /data/t1/' + subject + '.nii.gz \
        	--sid ' + subject + \
        	' --sd /data/recon-all \
            --surf_only --fsaparc --surfreg --no_cuda'
        )
    else:
        print('File ' + stat_file + ' already exists. Skipping.')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Perform segmentation.")
    mutex = parser.add_mutually_exclusive_group()

    mutex.add_argument("-s", "--subject", type=str, default=None,
                       help="Subject to be processed.")
    mutex.add_argument("-sl", "--subjects_list", type=str, default=None,
                       help="List of all subject to be processed.")
    parser.add_argument("-fs_seg", "--fastsurfer_segmentation", action="store_true",
                        help='Run fastsurfer segmentation.')
    parser.add_argument("-fs_surf", "--fastsurfer_surface", action="store_true",
                        help='Run fastsurfer surface recon.')
    parser.add_argument("-recon_dir", "--recon_dir", type=str, required=True,
                        help='Directory where to store recon-all outputs.')
    parser.add_argument("-t1_dir", "--t1_dir", type=str, required=True,
                        help='Directory containing input T1 images.')
    parser.add_argument("-n_jobs", "--n_jobs", type=int, default=1,
                        help="Number of parallel jobs. Default 1.")
    parser.add_argument("-n_cpus", "--n_cpus", type=int, default=1,
                        help="Number of CPUs used by docker container. Default 1.")
    args = parser.parse_args()


    if args.subject:
        subjects = np.array([args.subject])

    if args.subjects_list:
        with open(args.subjects_list, 'r') as f:
            subjects = np.array([subject.strip() for subject in f.readlines()])

    if args.fastsurfer_segmentation:
        for subject in subjects:
            fastsurfer_segmentation(subject, args.recon_dir, args.t1_dir)

    if args.fastsurfer_surface:
        params = [(subject, args.recon_dir, args.t1_dir, args.n_cpus)
                  for subject in subjects]
        with Pool(processes=args.n_jobs) as pool:
            pool.starmap(fastsurfer_surface, params)
