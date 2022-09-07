% Initialization
run_spm12;
spm('defaults','fmri');
spm_jobman('initcfg');

root_dir = pwd;
data_dir = [root_dir filesep 'data'];

for dataset = {'NP1', 'EMBARC'}

    % Load subjects

    if strcmp(dataset{1}, 'NP1')
        tbl = readtable([root_dir filesep 'tables/NP1.csv']);
        subjects = tbl.('subjects');
    end

    if strcmp(dataset{1}, 'EMBARC')
        tbl = readtable([root_dir filesep 'tables/EMBARC.csv']);
        subjects = tbl.('subjects');
    end

    %% Compute total intercranial volume

    vbm_dir = [data_dir filesep dataset{1} filesep 'vbm'];
    icv = [];

    for n = 1:numel(subjects)

        icv_file = [vbm_dir filesep subjects{n} '_tissuevolumes.csv'];

        if exist(icv_file, 'file') ~= 2

            % Compute ICV
            matlabbatch = {};
            matlabbatch{1}.spm.util.tvol.matfiles = {strcat([vbm_dir filesep], strcat(subjects{n}, '_seg8.mat'))};
            matlabbatch{1}.spm.util.tvol.tmax = 3;
            matlabbatch{1}.spm.util.tvol.mask = {'/usr/local/nru/spm12/tpm/mask_ICV.nii,1'};
            matlabbatch{1}.spm.util.tvol.outf = icv_file;
            spm_jobman('run',matlabbatch);

        end

        volumes = readmatrix(icv_file);
        icv(n) = sum(volumes(2:4)); %#ok<SAGROW>

    end

    %% Define statistical model

    for criteria = {'Responder', 'Remitter'}

        if strcmp(criteria, 'Responder')
            group = cellfun(@(x) strcmp(x, criteria{1}), tbl.('responder_hamd6'));
        end

        if strcmp(criteria, 'Remitter')
            group = cellfun(@(x) strcmp(x, criteria{1}), tbl.('remitter_hamd6'));
        end

        age = tbl.('age');
        sex = tbl.('sex');
        sex = cellfun(@(x) double(strcmp(x, 'Male')), sex);

        matlabbatch = {};
        matlabbatch{1}.spm.stats.factorial_design.dir = {[vbm_dir filesep criteria{1} '_results']};
        matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = strcat(strcat([vbm_dir filesep], strcat('smwc1', subjects(group)), '.nii,1'));
        matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = strcat(strcat([vbm_dir filesep], strcat('smwc1', subjects(~group)), '.nii,1'));
        matlabbatch{1}.spm.stats.factorial_design.des.t2.dept = 0;
        matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1;
        matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca = 0;
        matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova = 0;
        matlabbatch{1}.spm.stats.factorial_design.cov(1).c = [icv(group) icv(~group)];
        matlabbatch{1}.spm.stats.factorial_design.cov(1).cname = 'ICV';
        matlabbatch{1}.spm.stats.factorial_design.cov(1).iCFI = 1;
        matlabbatch{1}.spm.stats.factorial_design.cov(1).iCC = 5;
        matlabbatch{1}.spm.stats.factorial_design.cov(2).c = [age(group); age(~group)];
        matlabbatch{1}.spm.stats.factorial_design.cov(2).cname = 'Age';
        matlabbatch{1}.spm.stats.factorial_design.cov(2).iCFI = 1;
        matlabbatch{1}.spm.stats.factorial_design.cov(2).iCC = 5;
        matlabbatch{1}.spm.stats.factorial_design.cov(3).c = [sex(group); sex(~group)];
        matlabbatch{1}.spm.stats.factorial_design.cov(3).cname = 'Sex';
        matlabbatch{1}.spm.stats.factorial_design.cov(3).iCFI = 1;
        matlabbatch{1}.spm.stats.factorial_design.cov(3).iCC = 5;
        matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
        matlabbatch{1}.spm.stats.factorial_design.masking.tm.tma.athresh = 0.1;
        matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
        matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
        matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
        matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
        matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
        matlabbatch{2}.spm.stats.fmri_est.spmmat = {[vbm_dir filesep criteria{1} '_results' filesep 'SPM.mat']};
        matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
        matlabbatch{3}.spm.stats.con.spmmat = {[vbm_dir filesep criteria{1} '_results' filesep 'SPM.mat']};
        matlabbatch{3}.spm.stats.con.consess{1}.fcon.name = [criteria{1} ' > or < Non-' criteria{1}];
        matlabbatch{3}.spm.stats.con.consess{1}.fcon.weights = eye(2) - 0.5;
        matlabbatch{3}.spm.stats.con.consess{1}.fcon.sessrep = 'none';
        matlabbatch{3}.spm.stats.con.delete = 1;
        matlabbatch{4}.spm.stats.results.spmmat = {[vbm_dir filesep criteria{1} '_results' filesep 'SPM.mat']};
        matlabbatch{4}.spm.stats.results.conspec.titlestr = '';
        matlabbatch{4}.spm.stats.results.conspec.contrasts = Inf;
        matlabbatch{4}.spm.stats.results.conspec.threshdesc = 'FWE';
        matlabbatch{4}.spm.stats.results.conspec.thresh = 0.0500;
        matlabbatch{4}.spm.stats.results.conspec.extent = 0;
        matlabbatch{4}.spm.stats.results.conspec.conjunction = 1;
        matlabbatch{4}.spm.stats.results.conspec.mask.image.name = {[vbm_dir filesep criteria{1} '_results' filesep 'mask.nii,1']};
        matlabbatch{4}.spm.stats.results.conspec.mask.image.mtype = 0;
        matlabbatch{4}.spm.stats.results.units = 1;
        matlabbatch{4}.spm.stats.results.export{1}.pdf = 1;

        spm_jobman('run',matlabbatch);

    end
end
