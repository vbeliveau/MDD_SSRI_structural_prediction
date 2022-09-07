#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from os.path import join

root_dir  = pwd
data_dir  = join(root_dir, 'data')

os.chdir(root_dir)


for fs_version in ['fastsurfer']:
    for dataset in ['NP1', 'EMBARC']:

        recon_dir = None
        subjects = None

        recon_dir = join(data_dir, dataset, 'recon-all', fs_version)

        suffix = ''  # to be appended to subejct id
        if dataset == 'NP1':
            df = pd.read_excel(join(root_dir, 'lists', 'MR_NP1_HC_DBPROJECT_Vincent.xlsx'))
            df.rename(columns = {'RH-MR Lognumber': 'subjects',
                                 'Person status': 'status',
                                 'Documented compliance at week 8?': 'compliance_week8'},
                      inplace=True)
            df.query('(status == \'Case\') and (compliance_week8 == \'Yes\')', inplace=True)
            subjects = df.subjects.to_numpy()
            suffix = '_GD'

        if dataset == 'EMBARC':
            with open(join(data_dir, 'EMBARC', 'embarc_subjects'), 'r') as f:
                subjects = np.array([subject.strip() for subject in f.readlines()])

        if recon_dir is None or subjects is None:
            raise ValueError('FreeSurfer version ' + fs_version + ' is not implemented for dataset ' + dataset)

        aparc_col_names = 'StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd'.split(' ')
        aseg_col_names = 'Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange'.split(' ')

        subcort_regions = [
                'Lateral-Ventricle',
                'Cerebellum-Cortex',
                'Thalamus-Proper',
                'Caudate',
                'Putamen',
                'Pallidum',
                'Hippocampus',
                'Amygdala',
                'Accumbens-area'
            ]

        df = pd.DataFrame({'subjects': subjects})

        # Append thicknes data
        for hemi in ['lh', 'rh']:

            regions = None
            thickness = []
            surface_area = []

            for subject in subjects:

                # Load stat file
                stat_file = join(recon_dir, subject + suffix, 'stats', hemi + '.aparc.stats')
                with open(stat_file) as f:
                    lines = np.array(f.readlines())

                # Mean thickness
                mean_thickness = \
                    float(lines[[l.startswith('# Measure Cortex, MeanThickness')
                                          for l in lines]][0].split(' ')[-2][:-1])

                # Mean surface area
                mean_surface_area = \
                    float(lines[[l.startswith('# Measure Cortex, WhiteSurfArea')
                                        for l in lines]][0].split(' ')[-2][:-1])

                # Clean up stats
                lines = list(filter(lambda x: not x.startswith('#'), lines))
                lines = [list(filter(lambda x: x != '', l.strip().split(' '))) for l in lines]
                df_ = pd.DataFrame(np.vstack(lines), columns=aparc_col_names)

                # Make sur we have column headers
                if regions is None:
                    regions = df_['StructName']

                # Store data
                thickness += [list(df_['ThickAvg']) + [mean_thickness]]
                surface_area += [list(df_['SurfArea']) + [mean_surface_area]]

            columns = \
                ['.'.join([hemi, region, 'thickness']) for region in list(regions) + ['whole_hemisphere']] + \
                ['.'.join([hemi, region, 'surface_area']) for region in list(regions) + ['whole_hemisphere']]

            df_ = pd.DataFrame(
                    np.hstack((np.vstack(thickness), np.vstack(surface_area))),
                    columns=columns,
                    dtype=float
                 )

            df = pd.concat([df, df_], axis=1)

        # Append volumetric data
        columns = None
        volumes = []
        for subject in subjects:

            # Extract ICV
            icv_file = join(data_dir, dataset, 'ICV', subject + suffix, 'icv.txt')
            with open(icv_file) as f:
                lines = np.array(f.readlines())
            icv = float(lines[0].split(' ')[1])

            # Extract volumes

            # Load stat file
            stat_file = join(recon_dir, subject + suffix, 'stats', 'aseg.stats')
            with open(stat_file) as f:
                lines = np.array(f.readlines())

            # Clean up stats
            lines = list(filter(lambda x: not x.startswith('#'), lines))
            lines = [list(filter(lambda x: x != '', l.strip().split(' '))) for l in lines]
            df_ = pd.DataFrame(np.vstack(lines), columns=aseg_col_names)
            keep_region = [region in ['Left-' + region_ for region_ in subcort_regions] or
                           region in ['Right-' + region_ for region_ in subcort_regions]
                           for region in df_['StructName']]
            df_ = df_.iloc[keep_region]

            if columns is None:
                regions = df_['StructName']
                columns = ['.'.join([region.replace('Left-', 'lh.').replace('Right-', 'rh.').lower(), 'volume'])
                           for region in list(regions) + ['ICV']]

            volumes += [list(df_['Volume_mm3']) + [icv]]

        df_ = pd.DataFrame(np.vstack(volumes), columns=columns, dtype=float)
        df = pd.concat([df, df_], axis=1)

        # Save data
        csv_file = join(data_dir, dataset, 'freesurfer_measures.' + fs_version + '.csv')
        print('Saving data to ' + csv_file)
        df.to_csv(csv_file, index = False)
