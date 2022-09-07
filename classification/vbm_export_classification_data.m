function VBM_export_classification_data(dataset, criteria, training, test, out_dir)

    dataset = char(dataset);
    criteria = char(criteria);
    training = cellfun(@char, training, 'UniformOutput', false);
    test = cellfun(@char, test, 'UniformOutput', false);
    out_dir = char(out_dir);

    % Initialization
    run_spm12;
    spm('defaults','fmri');
    spm_jobman('initcfg');

    root_dir = pwd;
    data_dir = [root_dir filesep 'data'];
    vbm_dir = [data_dir filesep dataset filesep 'vbm'];

    if strcmp(dataset, 'NP1')
        tbl = readtable([root_dir filesep 'tables/NP1.csv']);
    end

    if strcmp(dataset, 'EMBARC')
        tbl = readtable([root_dir filesep 'tables/EMBARC.csv']);
    end

    %% Compute total intercranial volume

    icv = [];

    for n = 1:numel(training)

        icv_file = [vbm_dir filesep training{n} '_tissuevolumes.csv'];

        if exist(icv_file, 'file') ~= 2

            % Compute ICV
            matlabbatch = {};
            matlabbatch{1}.spm.util.tvol.matfiles = {strcat([vbm_dir filesep], strcat(training{n}, '_seg8.mat'))};
            matlabbatch{1}.spm.util.tvol.tmax = 3;
            matlabbatch{1}.spm.util.tvol.mask = {'/usr/local/nru/spm12/tpm/mask_ICV.nii,1'};
            matlabbatch{1}.spm.util.tvol.outf = icv_file;
            spm_jobman('run',matlabbatch);

        end

        volumes = readmatrix(icv_file);
        icv(n) = sum(volumes(2:4));  %#ok<AGROW>

    end

    %% Define statistical model

    training_index = cellfun(@(x) find(strcmp(tbl.('subjects'), x)), training);

    if strcmp(criteria, 'Responder')
        group = cellfun(@(x) strcmp(x, criteria), tbl.('responder_hamd6'));
        group = group(training_index);
    end

    if strcmp(criteria, 'Remitter')
        group = cellfun(@(x) strcmp(x, criteria), tbl.('remitter_hamd6'));
        group = group(training_index);
    end

    age = tbl.('age');
    age = age(training_index);
    sex = tbl.('sex');
    sex = cellfun(@(x) double(strcmp(x, 'Male')), sex);
    sex = sex(training_index);

    matlabbatch = {};
    matlabbatch{1}.spm.stats.factorial_design.dir = {out_dir};
    matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = cellstr(strcat(strcat([vbm_dir filesep], strcat('smwc1', training(group)), '.nii,1')));
    matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = cellstr(strcat(strcat([vbm_dir filesep], strcat('smwc1', training(~group)), '.nii,1')));
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
    matlabbatch{2}.spm.stats.fmri_est.spmmat = {[out_dir filesep 'SPM.mat']};
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
    matlabbatch{3}.spm.stats.con.spmmat = {[out_dir filesep 'SPM.mat']};
    matlabbatch{3}.spm.stats.con.consess{1}.fcon.name = [criteria ' > or < Non-' criteria];
    matlabbatch{3}.spm.stats.con.consess{1}.fcon.weights = eye(2) - 0.5;
    matlabbatch{3}.spm.stats.con.consess{1}.fcon.sessrep = 'none';
    matlabbatch{3}.spm.stats.con.delete = 1;
    matlabbatch{4}.spm.stats.results.spmmat = {[out_dir filesep 'SPM.mat']};
    matlabbatch{4}.spm.stats.results.conspec.titlestr = '';
    matlabbatch{4}.spm.stats.results.conspec.contrasts = Inf;
    matlabbatch{4}.spm.stats.results.conspec.threshdesc = 'none';
    matlabbatch{4}.spm.stats.results.conspec.thresh = 0.005;
    matlabbatch{4}.spm.stats.results.conspec.extent = 0;
    matlabbatch{4}.spm.stats.results.conspec.conjunction = 1;
    matlabbatch{4}.spm.stats.results.conspec.mask.image.name = {[out_dir filesep 'mask.nii,1']};
    matlabbatch{4}.spm.stats.results.conspec.mask.image.mtype = 0;
    matlabbatch{4}.spm.stats.results.units = 1;
    matlabbatch{4}.spm.stats.results.export{1}.binary.basename = 'classification';

    spm_jobman('run',matlabbatch);

    %% Extract VBM values from significant clusters

    % Load mask
    mask = spm_read_vols(spm_vol([out_dir filesep 'spmF_0001_classification.nii'])) == 1;

    % Extact VBM values
    X_train = [];
    for n = 1:numel(training)
        data = spm_read_vols(spm_vol([vbm_dir filesep 'smwc1' training{n}, '.nii']));
        X_train = [X_train; data(mask)']; %#ok<AGROW>
    end

    dlmwrite([out_dir filesep 'X_train.csv'], X_train, 'precision', 10)

    X_test = [];
    for n = 1:numel(test)
        data = spm_read_vols(spm_vol([vbm_dir filesep 'smwc1' test{n}, '.nii']));
        X_test = [X_test; data(mask)']; %#ok<AGROW>
    end

    dlmwrite([out_dir filesep 'X_test.csv'], X_test, 'precision', 10)

end
