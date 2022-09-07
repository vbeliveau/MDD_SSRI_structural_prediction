run_spm12;

root_dir = pwd;

for dataset = {'NP1', 'EMBARC'}

    % Load subjects

    if strcmp(dataset, 'NP1')
        tbl = readtable([root_dir filesep 'tables/NP1.csv']);
        subjects = tbl.('subjects');
    end

    if strcmp(dataset, 'EMBARC')
        tbl = readtable([root_dir filesep 'tables/EMBARC.csv']);
        subjects = tbl.('subjects');
    end

    icv_dir = [root_dir '/data/' dataset{1} '/ICV'];
    unix(['mkdir -p ' icv_dir]);

    % Define path to T1 directory
    t1_dir = [root_dir '/data/' dataset{1} '/t1'];

    for i = 1:length(subjects)

        subject = subjects{i};
        out_dir = [icv_dir filesep subject];

        % Check if subject is already processed
        icv_file = [out_dir filesep 'icv.txt'];

        if exist(icv_file, 'file') ~= 2

            disp(['Processing ' subject]);

            % Create output directory
            unix(['mkdir -p ' out_dir]);

            % Convert nii.gz to nii
            if strcmp(dataset, 'NP1')
                nii_gz_file = [t1_dir filesep subject '_GD.nii.gz'];
            else
                nii_gz_file = [t1_dir filesep subject '.nii.gz'];
            end
            nii_file = [out_dir filesep subject '.nii'];
            unix(['mri_convert ' nii_gz_file ' ' nii_file]);

            % move to subject data folder
            cd(out_dir);

            % run matlabbatch job
            try

                matlabbatch = {};
                matlabbatch{1}.spm.spatial.preproc.channel.vols = {[nii_file ',1']};
                matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
                matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
                matlabbatch{1}.spm.spatial.preproc.channel.write = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {'/usr/local/nru/spm12/tpm/TPM.nii,1'};
                matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 2;
                matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {'/usr/local/nru/spm12/tpm/TPM.nii,2'};
                matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 2;
                matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {'/usr/local/nru/spm12/tpm/TPM.nii,3'};
                matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
                matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {'/usr/local/nru/spm12/tpm/TPM.nii,4'};
                matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 4;
                matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {'/usr/local/nru/spm12/tpm/TPM.nii,5'};
                matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
                matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {'/usr/local/nru/spm12/tpm/TPM.nii,6'};
                matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
                matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
                matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
                matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
                matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
                matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
                matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
                matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
                matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];
                matlabbatch{2}.spm.util.tvol.matfiles(1) = {[out_dir filesep subject '_seg8.mat']};
                matlabbatch{2}.spm.util.tvol.tmax = 3;
                matlabbatch{2}.spm.util.tvol.mask = {'/usr/local/nru/spm12/tpm/mask_ICV.nii,1'};
                matlabbatch{2}.spm.util.tvol.outf = 'tissuevolumes';

                spm('defaults','fmri');
                spm_jobman('initcfg');
                spm_jobman('run',matlabbatch);

                % write out volumes
                csv_file = [out_dir filesep 'tissuevolumes.csv'];
                volumes = readmatrix(csv_file);
                icv = sum(volumes(2:4));
                f = fopen(icv_file,'w');
                fprintf(f,'ICV %f', icv);
                fclose(f);

            catch err % if there's an error, take notes & move on
                disp(['ERROR: ' files(i).name]);
                continue;
            end
        end

    end
end
