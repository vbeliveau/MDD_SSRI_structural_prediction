% Initialization
run_spm12;
spm('defaults','fmri');
spm_jobman('initcfg');

root_dir = pwd;
data_dir = [root_dir filesep 'data'];

tbl = readtable([root_dir filesep 'tables/NP1.csv']);
NP1_subjects = tbl.('subjects');

tbl = readtable([root_dir filesep 'tables/EMBARC.csv']);
EMBARC_subjects = tbl.('subjects');

%% Segment images

for dataset = {'NP1', 'EMBARC'}

    % Load subjects

    if strcmp(dataset{1}, 'NP1')
        subjects = NP1_subjects;
    end

    if strcmp(dataset{1}, 'EMBARC')
        tbl = readtable([root_dir filesep 'tables/EMBARC.csv']);
        subjects = EMBARC_subjects;
    end

    % Define output directory for VM results
    vbm_dir = [data_dir filesep dataset{1} filesep 'vbm'];
    unix(['mkdir -p ' vbm_dir]);

    t1_dir = [data_dir filesep dataset{1} filesep 't1'];

    for i = 1:numel(subjects)

        subject = subjects{i};
        nii_file = [vbm_dir filesep subject '.nii'];

        % Convert files from nii.gz to nii

        if exist(nii_file, 'file') ~= 2

            % Convert nii.gz to nii
            if strcmp(dataset, 'NP1')
                nii_gz_file = [t1_dir filesep subject '_GD.nii.gz'];
            else
                nii_gz_file = [t1_dir filesep subject '.nii.gz'];
            end

            unix(['mri_convert ' nii_gz_file ' ' nii_file]);

        end

        % Segment images

        if exist([vbm_dir filesep 'c1' subjects{n} '.nii'], 'file') ~= 2

            matlabbatch = {};
            matlabbatch{1}.spm.spatial.preproc.channel.vols = {[vbm_dir filesep subjects{n} '.nii']};
            matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 1e-3;
            matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
            matlabbatch{1}.spm.spatial.preproc.channel.write = [0 0];
            matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {'/usr/local/nru/spm12/tpm/TPM.nii,1'};
            matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
            matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 1];
            matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
            matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {'/usr/local/nru/spm12/tpm/TPM.nii,2'};
            matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
            matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 1];
            matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
            matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {'/usr/local/nru/spm12/tpm/TPM.nii,3'};
            matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
            matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
            matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
            matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {'/usr/local/nru/spm12/tpm/TPM.nii,4'};
            matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
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

        %     save(['segment' dataset '.mat'], 'matlabbatch')

            spm_jobman('run',matlabbatch);

        end
    end
end

%% Dartel create template for NP1 dataset

vbm_dir = [data_dir filesep 'NP1' filesep 'vbm'];

if exist([vbm_dir filesep 'Template_6.nii'], 'file') ~= 2

    subjects = NP1_subjects;

    rc1_files = strcat(strcat([vbm_dir filesep], strcat('rc1', subjects)), '.nii,1');
    rc2_files = strcat(strcat([vbm_dir filesep], strcat('rc2', subjects)), '.nii,1');

    matlabbatch = {};
    matlabbatch{1}.spm.tools.dartel.warp.images = {rc1_files, rc2_files};
    matlabbatch{1}.spm.tools.dartel.warp.settings.template = 'Template';
    matlabbatch{1}.spm.tools.dartel.warp.settings.rform = 0;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).rparam = [4 2 1e-6];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).K = 0;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).slam = 16;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).rparam = [2 1 1e-6];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).K = 0;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).slam = 8;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).rparam = [1 0.5 1e-6];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).K = 1;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).slam = 4;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).rparam = [0.5 0.25 1e-6];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).K = 2;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).slam = 2;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).rparam = [0.25 0.125 1e-6];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).K = 4;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).slam = 1;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).rparam = [0.25 0.125 1e-6];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).K = 6;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).slam = 0.5;
    matlabbatch{1}.spm.tools.dartel.warp.settings.optim.lmreg = 0.0100;
    matlabbatch{1}.spm.tools.dartel.warp.settings.optim.cyc = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.optim.its = 3;

    % save('dartel_create_template.mat', 'matlabbatch')

    spm_jobman('run',matlabbatch);

end

%% Align EMBARC to Dartel template

subjects = EMBARC_subjects;
vbm_dir = [data_dir filesep 'EMBARC' filesep 'vbm'];

for n = 1:numel(subjects)

    if exist([vbm_dir filesep 'u_rc1' subjects{n} '.nii'], 'file') ~= 2

        rc1_file = {[vbm_dir filesep 'rc1', subjects{n} '.nii,1']};
        rc2_file = {[vbm_dir filesep 'rc2', subjects{n} '.nii,1']};

        matlabbatch = {};
        matlabbatch{1}.spm.tools.dartel.warp1.images = {rc1_file, rc2_file};
        matlabbatch{1}.spm.tools.dartel.warp1.settings.rform = 0;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(1).its = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(1).rparam = [4 2 1e-6];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(1).K = 0;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(1).template{1} = [data_dir filesep 'NP1' filesep 'vbm' filesep 'Template_1.nii'];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(2).its = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(2).rparam = [2 1 1e-6];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(2).K = 0;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(2).template{1} = [data_dir filesep 'NP1' filesep 'vbm' filesep 'Template_2.nii'];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(3).its = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(3).rparam = [1 0.5 1e-6];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(3).K = 1;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(3).template{1} = [data_dir filesep 'NP1' filesep 'vbm' filesep 'Template_3.nii'];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(4).its = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(4).rparam = [0.5 0.25 1e-6];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(4).K = 2;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(4).template{1} = [data_dir filesep 'NP1' filesep 'vbm' filesep 'Template_4.nii'];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(5).its = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(5).rparam = [0.25 0.125 1e-6];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(5).K = 4;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(5).template{1} = [data_dir filesep 'NP1' filesep 'vbm' filesep 'Template_5.nii'];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(6).its = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(6).rparam = [0.25 0.125 1e-6];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(6).K = 6;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.param(6).template{1} = [data_dir filesep 'NP1' filesep 'vbm' filesep 'Template_6.nii'];
        matlabbatch{1}.spm.tools.dartel.warp1.settings.optim.lmreg = 0.0100;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.optim.cyc = 3;
        matlabbatch{1}.spm.tools.dartel.warp1.settings.optim.its = 3;

        % save('dartel_create_template.mat', 'matlabbatch')

        spm_jobman('run', matlabbatch);

    end
end

%% Normalize to MNI space

u_files = {};
c1_files = {};

for dataset = {'NP1', 'EMBARC'}

    % Load subjects

    if strcmp(dataset{1}, 'NP1')
        subjects = NP1_subjects;
    end

    if strcmp(dataset{1}, 'EMBARC')
        tbl = readtable([root_dir filesep 'tables/EMBARC.csv']);
        subjects = EMBARC_subjects;
    end

    vbm_dir = [data_dir filesep dataset{1} filesep 'vbm'];

    for n = 1:numel(subjects)

        if exist([vbm_dir filesep 'smwc1' subjects{n} '.nii'], 'file') ~= 2

            if strcmp(dataset, 'NP1')
                subjects = NP1_subjects;
                vbm_dir = [data_dir filesep dataset{1} filesep 'vbm'];
                u_file = {[vbm_dir filesep 'u_rc1' subjects{n} '_Template.nii']};
            end

            if strcmp(dataset, 'EMBARC')
                subjects = EMBARC_subjects;
                u_file = {[vbm_dir filesep 'u_rc1' subjects{n} '.nii']};
            end

            c1_file = {[vbm_dir filesep 'c1' subjects{n} '.nii']};

            matlabbatch = {};
            matlabbatch{1}.spm.tools.dartel.mni_norm.template{1} = [data_dir filesep 'NP1' filesep 'vbm' filesep 'Template_6.nii'];
            matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs.flowfields = u_file;
            matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs.images{1} = c1_file;
            matlabbatch{1}.spm.tools.dartel.mni_norm.vox = [NaN NaN NaN];
            matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [NaN NaN NaN;
                                                           NaN NaN NaN];
            matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 1;
            matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [8 8 8];

            spm_jobman('run',matlabbatch);

        end
    end
end
