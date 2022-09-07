library(tidyr)
library(dplyr)
library(purrr)
library(stringr)
library(randomForest)
library(pROC)
library(splitTools)
library(caret)
library(parallel)
library(ggplot2)
library(ggpubr)
library(readxl)
library(xgboost)

# Utility functions for classification

fs.cort.clean.names = list(
  'bankssts' = 'Banks of the STS',
  'caudalanteriorcingulate' = 'Caudal anterior cingulate',
  'caudalmiddlefrontal' = 'Caudal middle frontal',
  'cuneus' = 'Cuneus',
  'entorhinal' = 'Entorhinal',       
  'frontalpole' = 'Frontal pole',
  'fusiform' = 'Fusiform',
  'inferiorparietal' = 'Inferior parietal',
  'inferiortemporal' = 'Inferior temporal',
  'insula' = 'Insula',
  'isthmuscingulate' = 'Isthmus of cingulate',
  'lateraloccipital' = 'Lateral occipital',
  'lateralorbitofrontal' = 'Lateral orbitofrontal',
  'lingual' = 'Lingual',
  'medialorbitofrontal' = 'Medial orbitofrontal',
  'middletemporal' = 'Middle temporal',
  'paracentral' = 'Paracentral',
  'parahippocampal' = 'Parahippocampal',
  'parsopercularis' = 'Pars opercularis', 
  'parsorbitalis' = 'Pars orbitalis',
  'parstriangularis' = 'Pars triangularis',
  'pericalcarine' = 'Pericalcarine',
  'postcentral' = 'Postcentral',
  'posteriorcingulate' = 'Posterior cingulate',
  'precentral' = 'Precentral',
  'precuneus' = 'Precuneus',
  'rostralanteriorcingulate' = 'Rostral anterior cingulate',
  'rostralmiddlefrontal' = 'Rostral middle frontal',
  'superiorfrontal' = 'Superior frontal',
  'superiorparietal' = 'Superior parietal',
  'superiortemporal' = 'Superior temporal',
  'supramarginal' = 'Supramarginal',
  'temporalpole' = 'Temporal pole',
  'transversetemporal' = 'Tranverse temporal',
  'whole_hemisphere' = 'Mean Thickness'
)

fs.subcort.clean.names = list(
  'accumbens.area' = 'Accumbens area',
  'amygdala' = 'Amygdala',
  'caudate' = 'Caudate',
  'cerebellum.cortex' = 'Cerebellum cortex',
  'hippocampus' = 'Hippocampus',
  'lateral.ventricle' = 'Lateral ventricle',
  'pallidum' = 'Pallidum',
  'putamen' = 'Putamen',
  'thalamus.proper' = 'Thalamus'
)

fs.others.cort.clean.names = list(
  'lateral.ventricle' = 'Lateral ventricle',
  'mean_thick' = 'Mean Thickness'
)

fs.clean.names = c(
  fs.cort.clean.names,
  fs.subcort.clean.names
  # fs.others.clean.names
)


other.clean.names = list(
  'age' = 'Age',
  'hamd6.base' = 'HAMD-6 Baseline',
  'single.recurrent' = 'Recurrence status',
  'sex' = 'Sex'
)

all.clean.names = c(fs.clean.names, other.clean.names)


clean_name = function(name) {
  
  name = str_replace(name, 'lh.', 'Left ')
  name = str_replace(name, 'rh.', 'Right ')
  name = str_replace(name, '.thickness', '')
  name = str_replace(name, '.volume', '')
  
  for (pattern in names(fs.clean.names)) {
    name = str_replace(name, pattern, fs.clean.names[[pattern]])
  }
  
  name
  
}


load_data = function(
    dataset = NULL,
    criteria = NULL,
    hamd = NULL,
    include.vars = NULL,
    include.freesurfer = F,
    freesurfer.measures = NULL,
    average.lr = T,
    keep.subjects = F
  ) {
  
  # Load FreeSufer measures
  if (include.freesurfer) {
    
    df.fs = read.csv(file.path('data', dataset, 'freesurfer_measures.fastsurfer.csv'))
    
    # Select measures to include
    df.fs = df.fs %>% select(subjects, ends_with(freesurfer.measures))
    
    # Average left/right regions
    if (average.lr) {
      
      # Select regions
      regions = c()
      if ('thickness' %in% freesurfer.measures)
        regions = c(regions, paste(names(fs.cort.clean.names), 'thickness', sep = '.'))
      if ('surface_area' %in% freesurfer.measures)
        regions = c(regions, paste(names(fs.cort.clean.names), 'surface_area', sep = '.'))
      if ('volume' %in% freesurfer.measures)
        regions = c(regions, paste(names(fs.subcort.clean.names), 'volume', sep = '.'))
      
      # Do averaging
      for (region in regions) {
        df.fs[,paste0('mean.', region)] =
          (df.fs[,paste0('lh.', region)] + df.fs[,paste0('rh.', region)])/2
      }
      
      # Remove merged regions
      df.fs = df.fs %>% select(-starts_with('lh'), -starts_with('rh'))
      
    }
    
  }
  
  # Extract demographics and clinical information 
  
  if (dataset == 'NP1') {
  
    # Load FreeSurfer measures

    df = read_excel(file.path('lists', 'MR_NP1_HC_DBPROJECT_Vincent.xlsx'))
    
    # Rename
    df = df %>% rename(
      subjects='RH-MR Lognumber',
      sex='Gender',
      single.recurrent='Single or recurrent MDD episode?',
      hamd6.base='HAMD-6 score - Baseline',
      hamd6.change.week8='NP1 secondary outcome - Percent change in HAMD-6 at week 8 compared to baseline',
      hamd17.base='HAM-D17 score (Interview): Total sum (0-52)',
      hamd17.week8='HAM-D17 score (Week8): Total sum (0-52)'
    )
    
    # Compute age at MR scan
    df$age = as.numeric(difftime(df$`RH-MR scan date`, df$`Date of birth`, unit = "weeks")) / 52.25
    
    # Convert HAMD-6 baseline for demographics table
    df$hamd6.base = as.integer(df$hamd6.base)
    
    # Compute HAMD-6 at week 8
    df$hamd6.week8 = as.integer(df$hamd6.base + df$hamd6.base * df$hamd6.change.week8/100)

    # Compute change in HAMD-17 at week 8
    df$hamd17.change.week8 = (df$hamd17.week8 - df$hamd17.base)/df$hamd17.base*100
    
    # Filter data
    df = df %>%
      filter(!is.na(hamd6.change.week8) &
             `Documented compliance at week 8?` == 'Yes')
        
    # Convert variables to factor
    df$sex = factor(df$sex, levels = c('Female', 'Male'))
    df$single.recurrent = factor(df$single.recurrent,
                                  levels = c('Single', 'Recurrent'))
    
  }

  if (dataset == 'EMBARC') {

    # Extract subjects
    
    # Load unblinded data and select subjects  
    df = read_excel(file.path('data', 'EMBARC', 'CT_Unblinded_Stage12.xlsx')) %>%
      rename(subjects = ProjectSpecificId,
             hamd17.base = w0_score_17,
             hamd17.week8 = w8_score_17)

    # Find out which subject had MRI at week 0
    df.mri = read.csv(file.path('data', 'EMBARC', 'Embarc All Subjects List 708.csv')) %>%
      rename(subjects = ProjectSpecificId)
    
    # Load additional clinical variables
    df.scid = read.csv(file.path('data', 'EMBARC', 'scid01_noheader.txt'), sep = '\t') %>%
      rename(subjects = src_subject_id)
    
    # Join dataset
    df = left_join(df, df.mri, by = 'subjects')
    df = left_join(df, df.scid, by = 'subjects')
    
    # Recode age & sex
    df$age = as.integer(df$age)
    df$sex = factor(df$gender, levels = c('Female', 'Male'))

    # Single/recurrent
    df$single.recurrent = factor(df$mdcnumep > 1) %>%
      recode(`FALSE`='Single', `TRUE`='Recurrent')
    
    # Assign HAMD-17 scores
    df$hamd17.base = as.integer(df$hamd17.base)
    df$hamd17.week8 = as.integer(df$hamd17.week8)
    df$hamd17.change.week8 = (df$hamd17.week8 - df$hamd17.base)/df$hamd17.base*100
    
    # Extract corresponding HAMD-6 scores
    
    df.hrsd = read.csv(
      file.path('data', 'EMBARC', 'hrsd01_noheader.txt'),
      sep = '\t'
    )

    df.hrsd$hamd6 = 
      df.hrsd$hmdsd + # 1. depressed mood
      df.hrsd$hvwsf + # 2. guilt feelings
      df.hrsd$hintr + # 7. work and activities
      df.hrsd$hslow + # 8. psychomotor retardation
      df.hrsd$hpanx + # 10. psychic anxiety
      df.hrsd$hamd_18 # 13. general somatic symptoms
    
    # Filter data by subjects and week
    df.hrsd = df.hrsd %>%
      filter(week %in% c(0, 8)) %>%
      select(src_subject_id, hamd6, week) %>%
      distinct() %>%
      pivot_wider(values_from = hamd6,
                  names_from = week,
                  names_prefix = 'hamd6.week') %>%
      rename(subjects = src_subject_id,
             hamd6.base = hamd6.week0)
    
    # Concatenate HAMD data
    df  = left_join(df, df.hrsd, by = 'subjects')
    
    # Compute HAMD-6 change at week 8
    df$hamd6.change.week8 = (df$hamd6.week8 - df$hamd6.base)/df$hamd6.base*100
    
    df = df %>%
      filter(Stage1TX == 'SER' &
             w0MRIacqXfr != '' &
             !is.na(hamd6.week8) &
             hamd17.base > 17  # to match NP1 data
          )
    
  }

  # Define response and remission
  
  # HAMD-6
  df$responder.hamd6 = as.factor(df$hamd6.change.week8 <= -50) %>%
    dplyr::recode(`TRUE`='Responder', `FALSE`='Non-Responder')
  
  df$remitter.hamd6 = as.factor(df$hamd6.week8 < 5) %>%
    dplyr::recode(`TRUE`='Remitter', `FALSE`='Non-Remitter')
  
  # HAMD-17
  df$responder.hamd17 = as.factor(df$hamd17.change.week8 <= -50) %>%
    dplyr::recode(`TRUE`='Responder', `FALSE`='Non-Responder')
  
  df$remitter.hamd17 = as.factor(df$hamd17.week8 < 8) %>%
    dplyr::recode(`TRUE`='Remitter', `FALSE`='Non-Remitter')
  
  # Assign group for classification
  if (hamd == 'hamd6') {
    if (criteria == 'response') {
      df$group = as.factor(df$hamd6.change.week8 <= -50)
    }
    
    if (criteria == 'remission') {
      df$group = as.factor(df$hamd6.week8 < 5)
    }
  }
   
  if (hamd == 'hamd17') {
    if (criteria == 'response') {
      df$group = as.factor(df$hamd17.change.week8 <= -50)
    }
    
    if (criteria == 'remission') {
      df$group = as.factor(df$hamd17.week8 < 8)
    }
  }
  
  if ('group' %in% colnames(df)) {
    df$group = as.factor(df$group) %>%
      recode(`TRUE`='case', `FALSE`='control')
    include.vars = c(include.vars, 'group')
  }
  
  # Filter variables info
  df = df %>% select(subjects, all_of(include.vars))
  
  # Concatenate using subject ids
  if (include.freesurfer) {
    df = left_join(df, df.fs, by = 'subjects')
  }
  
  # Remove subjects
  if (!keep.subjects) {
    df = df %>% select(-subjects)
  }
  
  # Convert tibble to data.frame
  df = df %>% data.frame()
  
  df
  
}


n_digits = function(num) {
  
  # Output correctly formatted number of digits
  
  as.character(floor(log10(abs(num))) + 1)
}


class_weights = function(group) {
 
  list(
    'control' = 1 - sum(group == 'control')/length(group),
    'case' = 1 - sum(group ==  'case')/length(group)
  )
  
}


train_classifier = function(
  X, y,
  classifier = NULL,
  use.class.weight = T,
  bootstrap = F,
  permute = F,
  seed = 42
) {
  
  
  ### Train a given classifier ###

  set.seed(seed)

  # Bootstrap or permute data, used in final evaluation on full dataset
  if (bootstrap) {
    ind = sample(nrow(X), replace=T)
    X = X[ind,]
    y = y[ind]
  }
  
  if (permute) {
    y = sample(y, replace=F)
  }
  
  # Create folds
  folds = create_folds(
    y,
    k = 5,
    type = 'stratified',
    seed = seed
  )
  
  control = trainControl(
    index = folds,
    method = "cv",
    number = 5,
    classProbs=TRUE,
    summaryFunction = twoClassSummary
  )
  
  # Train classifier
  if (classifier == 'randomForest') {

    # Code adapted from https://rpubs.com/phamdinhkhanh/389752
    
    customRF = list(type = "Classification",
                    library = "randomForest",
                    loop = NULL)
    
    customRF$parameters = data.frame(
      parameter = c("mtry", "ntree"),
      class = rep("numeric", 2),
      label = c("mtry", "ntree")
    )

    customRF$grid <- function(x, y, len = NULL, search = "grid") {}
    
    if (use.class.weight) {
      classwt = class_weights(y)
    } else {
      classwt = NULL
    }

    customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs) {
      randomForest(
        x, y,
        mtry = param$mtry,
        ntree = param$ntree,
        classwt = classwt
      )
    }
    
    # Predict label
    customRF$predict = function(modelFit, newdata, preProc = NULL, submodels = NULL) {
      predict(modelFit, newdata)
    }
    
    # Predict prob
    customRF$prob = function(modelFit, newdata, preProc = NULL, submodels = NULL) {
      predict(modelFit, newdata, type = "prob")
    }
    
    customRF$sort = function(x) x[order(x[,1]),]
    customRF$levels = function(x) x$classes
    
    tunegrid = expand.grid(.mtry=c(2:5), .ntree=c(1000, 1500, 2000, 2500))
    
    fit = train(
        X, y,
        method = customRF, 
        metric = 'ROC', 
        trControl = control,
        tuneGrid = tunegrid
      )

  } else if (classifier == 'xgbTree') {
    
    if (use.class.weight) {
      scale_pos_weight = sum(y == 'control')/sum(y == 'case')
    }
    
    fit = train(
      X, y,
      method = 'xgbTree',
      metric = 'ROC',
      preProcess = c('center', 'scale'),
      trControl = control,
      verbosity = 0,
      tuneLength = 2,
      scale_pos_weight = scale_pos_weight
    )

  } else if (classifier == 'svmRadialWeights') {

    if (use.class.weight) {
      Weights = class_weights(y)
    } else {
      Weights = NULL
    }
    
    fit = train(
      X, y,
      method = 'svmRadialWeights',
      metric = 'ROC',
      preProcess = c('center', 'scale', 'pca'),
      trControl = control,
      tuneLength = 5,
      Weights = Weights
    )
    
  } else if (classifier == 'glmnet') {
    
    weights = unlist(class_weights(y)[y])

    if (use.class.weight) {
      weights = unlist(class_weights(y)[y])
    } else {
      weights = NULL
    }
    
    fit = train(
      X, y,
      method = "glmnet",
      family = 'binomial',
      type.measure = "class",
      metric = 'ROC',
      preProcess = c('center', 'scale'),
      trControl = control,
      tuneLength = 10,
      weights = weights
    )
    
  } else {
    stop(paste0('Classifier ', classifier, ' is not recognized.'))
  }
  
  fit
  
}


run_cv = function(
    data = NULL,
    classifier = NULL,
    n_folds = 10,
    permute = F,
    use.class.weight = T,
    seed = 42,
    verbose = F,
    checkpoint.dir = '',
    xargs = NULL
) {
  
  # Wrapper for CV
  
  # Check if we should make regular checkpoints
  if (checkpoint.dir != '') {
    dir.create(checkpoint.dir, recursive = T, showWarnings = F)
    out.rds = file.path(checkpoint.dir, sprintf('seed-%i.rds', seed))
  } else {
    out.rds = ''
  }
  
  if ( !file.exists(out.rds) ) {
  
    set.seed(seed)  # make reproducible
    y = data$group
  
    # If VBM is requested, assume subjects are provided
    if ('vbm.params' %in% names(xargs) & !is.null(xargs$vbm.params)) {
      include.vbm = T
      vbm.params = xargs$vbm.params
    } else {
      include.vbm = F
    }
  
    if (include.vbm) {
      subjects = data$subjects
      data = data %>% select(-subjects)
    }
    
    # Transform factors to dummy variables for classifiers that require it
    if (classifier != c('randomForest')) {
      X = model.matrix(group ~ ., data = data)
      X = X[,-1]  # remove intercept
    } else {
      X = data %>% select(-group)
    }
    
    # If requested, permute labels to evaluate null
    if (permute) {
      # made reproducible by setting set.seed above
      y = sample(y, length(y), replace=F)
    }
    
    # Create folds
    folds = create_folds(
      y,
      k = n_folds,
      type = 'stratified',
      shuffle = T,
      seed = seed
    )
    
    cv_results = tibble(
      best_tune = tibble(),
      pred = list(),
      prob = list(),
      y = list()
    )
    
    if (verbose) {
      print(sprintf('Starting CV - seed %i', seed))
      start.time = Sys.time()
    }
    
    for (n_fold in 1:n_folds) {
      
      if (verbose) {
        print(paste0('Fold ', n_fold))
        start.time.fold = Sys.time()
      }
        
      # Retrieve training index
      fold.str = sprintf(paste0('Fold%0.', n_digits(n_folds),'i'), n_fold)
      train = folds[[fold.str]]
      X_train = X[train,]
      X_test = X[-train,]
      
      # Handle VBM
      if (include.vbm) {
        
        out_dir = file.path(vbm.params$vbm.dir, sprintf('seed-%i_fold-%i', seed, n_fold))
        
        if (!file.exists(file.path(out_dir, 'X_train.csv')) | 
            !file.exists(file.path(out_dir, 'X_test.csv'))) {
          training_subjects = subjects[train]
          test_subjects = subjects[-train]
          vbm_data_on_demand(vbm.params$dataset, vbm.params$criteria, training_subjects, test_subjects, out_dir)
        }
        
        X_train_ = read.csv(file.path(out_dir, 'X_train.csv'), header = F)
        X_test_ = read.csv(file.path(out_dir, 'X_test.csv'), header = F)
        X_train = cbind(X_train, X_train_)
        X_test = cbind(X_test, X_test_)
        
        # Clean up
        if (vbm.params$clean) {
          unlink(out_dir, recursive = T)
        }
      }
  
      # Train classifier
      fit = train_classifier(
              X_train,
              y[train],
              classifier = classifier,
              use.class.weight = use.class.weight
            )
  
      # Estimate probabilities and predicted labels by best model
      prob = predict(fit, newdata = X_test, type = 'prob')[,'case']
      pred = predict(fit, newdata = X_test)
      
      # Transform best tuning parameters to tibble and save any extra information
      best_tune = tibble(fit$bestTune)
      
      if (classifier == 'randomForest') {
        best_tune = best_tune %>% 
          tibble::add_column(importance = list(importance(fit$finalModel)))
      }
      
      # Save out results
      cv_results = cv_results %>% tibble::add_row(
        best_tune = best_tune,
        pred = list(pred),
        prob = list(prob),
        y = list(y[-train])
      )
      
      if (verbose) {
        end.time.fold = Sys.time()
        print(end.time.fold - start.time.fold)
      }
      
    }
  
    if (verbose) {
      print('End CV')
      end.time = Sys.time()
      print(end.time - start.time)
    }
    
    # Save out general parameters
    cv_results$seed = seed
    
  }
  
  # Save out results, or load if they already exist
  if (out.rds != '') {
    if ( file.exists(out.rds) ) {
      cv_results = readRDS(file = out.rds)
    } else {
      print(out.rds)
      saveRDS(cv_results, file = out.rds)
    }
  }
    
  cv_results
  
}


parallel_nested_cv = function(
  data = NULL,
  classifier = NULL,
  n_repetitions = 50,
  n_folds = 10,
  use.class.weight = T,
  n_jobs = 1,
  checkpoint.dir = '',
  ...
) {
  
  # Wrapper for repeated CV

  start.time = Sys.time()
  
  seeds = 1:n_repetitions
  
  # xgbTree is parallelized by default, override n_jobs parameter
  if (classifier %in% c('xgbTree')) {
    n_jobs = 1
  }
  
  cv_results = mcmapply(
    run_cv,
    data = list(data),
    classifier = classifier,
    n_folds = n_folds,
    use.class.weight = use.class.weight,
    seed = seeds,
    verbose = T,
    checkpoint.dir = checkpoint.dir,
    xargs = list(list(...)),
    mc.cores = n_jobs,
    mc.silent = T,
    SIMPLIFY = F
  )
  
  print('Repeated CV')
  end.time = Sys.time()
  print(end.time - start.time)

  tibble(
    cv_results = cv_results,
    n_repetition = 1:n_repetitions
  )
  
}


nested_cv_wrapper = function(
    data = NULL,
    classifier = NULL,
    n_repetitions = 50,
    n_folds = 10,
    use.class.weight = T,
    n_jobs = 1,
    out.rds = NULL,
    permutation.dir = NULL,
    n_permutations = NA,
    ...
  ) {

  if (length(out.rds) != '' & !file.exists(out.rds)) {
    
    print(paste0(out.rds, ' does not exist.'))
    
    # Specify where to save checkpoints, if requested
    checkpoint.dir = ''
    if ('cv.checkpoint' %in% names(list(...))) {
      cv.checkpoint = list(...)$cv.checkpoint
      if (cv.checkpoint) {
        checkpoint.dir = paste0(substr(out.rds, 1, nchar(out.rds) - 4), '.checkpoints')
      }
    }

    cv_results = parallel_nested_cv(
      data = data,
      classifier = classifier,
      n_repetitions = n_repetitions,
      n_folds = n_folds,
      use.class.weight = use.class.weight,
      permute = F,
      checkpoint.dir = checkpoint.dir,
      n_jobs = n_jobs,
      ...
    )
    
    cat(out.rds)
    saveRDS(cv_results, file = out.rds)

  }
  
  if (permutation.dir != '') {
    
    print('Performing permutations.')
    
    dir.create(permutation.dir, recursive = T, showWarnings = F)

    # Compute permutation results
    for (n_permutation in 1:n_permutations) {
      
      out.perm.rds = file.path(
        permutation.dir, paste0('permutation-', n_permutation, '.rds'))
      
      group = data$group  # Store original groups
      
      if (!file.exists(out.perm.rds)) {
      
        print(paste0('Running permutation ', n_permutation))
      
        # Permute groups
        set.seed(n_permutation)
        data$group = sample(group, replace = F)
        
        cv_results = parallel_nested_cv(
          data = data,
          classifier = classifier,
          n_repetitions = n_repetitions,
          n_folds = n_folds,
          use.class.weight = use.class.weight,
          n_jobs = n_jobs,
          ...
        )
        
        # Store and save results
        print(out.perm.rds)
        saveRDS(cv_results, file = out.perm.rds)

      }
      
    }
  }
}

get_suffix = function(
  include.vars = F,
  include.freesurfer = F,
  average.lr = F,
  freesurfer.measures = NULL,
  include.vbm = F
) {
  
  file.suffix = NULL
  
  if (!is.null(include.vars)) {
    file.suffix = str_c(file.suffix, 'baseline', sep = '.')
  }
  
  if (include.freesurfer) {
    file.suffix = str_c(file.suffix, 'fastsurfer', sep = '.')
    if (average.lr) {
      file.suffix = str_c(file.suffix, 'average', sep = '.')
    } else {
      file.suffix = str_c(file.suffix, 'lr', sep = '.')
    }
    
    file.suffix = paste(c(file.suffix, freesurfer.measures), collapse = '.')
    
  }
  
  if (include.vbm) {
    file.suffix = str_c(file.suffix, 'vbm', sep = '.')
  }
  
  file.suffix
  
}


train_nested_cv = function(
    study.params = NULL,
    classifiers = NULL,
    compute.permutations = F,
    n_permutations = NA,
    do.plot.roc = F,
    plot.permutations = F,
    n_repetitions = 10,
    n_folds = 5,
    use.class.weight = T,
    n_jobs = 1,
    results.dir = 'cv_results',
    cv.checkpoint = F
) {
  
  ### General wrapper handling study specific aspects of CV ###

  results = tibble(
    classifier = character(),
    dataset = character(),
    hamd = character(),
    criteria = character(),
    include_vars = logical(),
    include_freesurfer = logical(),
    thickness = logical(),
    surface_area = logical(),
    volume = logical(),
    average_lr = logical(),
    mean_auc = numeric(),
    mean_balanced_accuracy = numeric(),
    mean_sensitivity = numeric(),
    mean_specificity = numeric(),
    sd_auc = numeric(),
    sd_balanced_accuracy = numeric(),
    sd_sensitivity = numeric(),
    sd_specificity = numeric()
  )
  
  for (params in study.params) {

    dataset = params$dataset
    include.vars = params$include.vars[[1]]
    include.freesurfer = params$include.freesurfer
    
    if (include.freesurfer) {
      freesurfer.measures = params$freesurfer.measures[[1]]
      average.lr = params$average.lr
    } else {
      freesurfer.measures = NULL
      average.lr = NULL
    }
    
    criteria = params$criteria
    hamd = params$hamd
    classifier = params$classifier

    # Is VBM requested?
    if ('include.vbm' %in% colnames(params)){
      include.vbm = params$include.vbm
      if (include.vbm) {
        keep.subjects = T
      }      
    } else {
      include.vbm = F
      keep.subjects = F
    }
    
    file.suffix = get_suffix(
      include.vars = include.vars,
      include.freesurfer = include.freesurfer,
      average.lr = average.lr,
      freesurfer.measures = freesurfer.measures,
      include.vbm = include.vbm
    )

    # Pass on VBM parameters, if requested
    if (include.vbm){

      vbm.params = list(
        vbm.dir = file.path(getwd(), 'data', dataset, results.dir, file.suffix),
        dataset = dataset,
        criteria = criteria,
        clean = T
      )
      
    } else {
      vbm.params = NULL
    }
    
    # Load and format data
    data = load_data(
      dataset = dataset,
      criteria = criteria,
      hamd=hamd,
      include.vars = include.vars,
      include.freesurfer = include.freesurfer,
      freesurfer.measures = freesurfer.measures,
      average.lr = average.lr,
      keep.subjects = keep.subjects
    )
    
    # Assert results directories and specify output file
    dir.create(file.path(getwd(), 'data', dataset, results.dir),
               recursive = T, showWarnings = F)
    out.rds = file.path('data', dataset, results.dir, 
                        paste(classifier, criteria, hamd, file.suffix, 'rds', sep = '.'))        
    print(out.rds)
        
    if (compute.permutations | plot.permutations) {
      permutation.dir = file.path('data', dataset, results.dir, 
        paste('permutations', classifier, criteria, hamd, file.suffix, sep = '.'))
    } else {
      permutation.dir = ''
    }
    
    nested_cv_wrapper(
      data,
      classifier = classifier,
      n_repetitions = n_repetitions,
      n_folds = n_folds,
      use.class.weight = use.class.weight,
      n_jobs = n_jobs,
      out.rds = out.rds,
      permutation.dir = permutation.dir,
      n_permutations = n_permutations,
      vbm.params = vbm.params,
      cv.checkpoint = cv.checkpoint
    )
   
    df = readRDS(file = out.rds)
    metrics = bind_rows(df$cv_results %>% lapply(evaluate_metrics))
    
    # Store results
    results = results %>% add_row(tibble(
      classifier = classifier,
      dataset = dataset,
      hamd = hamd,
      criteria = criteria,
      include_vars = is.null(include.vars),
      include_freesurfer = include.freesurfer,
      thickness = ifelse(include.freesurfer, 'thickness' %in% freesurfer.measures, F),
      surface_area = ifelse(include.freesurfer, 'surface_area' %in% freesurfer.measures, F),
      volume = ifelse(include.freesurfer, 'volume' %in% freesurfer.measures, F),
      average_lr = average.lr,
      mean_auc = mean(metrics$auc),
      mean_balanced_accuracy = mean(metrics$balanced_accuracy),
      mean_sensitivity = mean(metrics$sensitivity),
      mean_specificity = mean(metrics$specificity),
      sd_auc = sd(metrics$auc),
      sd_balanced_accuracy = sd(metrics$balanced_accuracy),
      sd_sensitivity = sd(metrics$sensitivity),
      sd_specificity = sd(metrics$specificity)
    ))
    
    write.csv(
      results,
      file.path('data', dataset, results.dir, 'metrics.csv'),
      row.names = F
    )
    
    if (do.plot.roc) {
      
      dir.create(file.path('data', dataset, results.dir, 'figures'),
                 recursive = T, showWarnings = F)
      out.roc = file.path('data', dataset, results.dir, 'figures',
                    paste(classifier, hamd, file.suffix, 'roc', sep='.'))
      
      g = plot_cv_roc_wrapper(
        out.roc = out.roc,
        cv.rds = out.rds,
        permutation.dir = permutation.dir,
        title.prefix = paste(classifier,'-', str_to_title(criteria))
      )
      
      ggsave(
        paste0(out.roc, '.png'),
        plot = g,
        width=90, height=100,
        dpi=600,
        units = 'mm',
        bg = 'white'
      )

    }
  }
}


train_final_models = function(
  dataset = 'NP1',
  study.params = NULL,
  classifiers = NULL,
  n_bootstrap = 1,
  use.class.weight = T,
  n_jobs = 1,
  results.dir = 'final_models'
) {
  
  dir.create(file.path('data', dataset, results.dir),
             recursive = T, showWarnings = F)
  
  for (params in study.params) {
    
    dataset = params$dataset
    include.vars = params$include.vars[[1]]
    include.freesurfer = params$include.freesurfer
    freesurfer.measures = params$freesurfer.measures[[1]]
    average.lr = params$average.lr
    criteria = params$criteria
    hamd = params$hamd
    classifier = params$classifier

    # xgbTree is parallelized by default, override n_jobs parameter
    if (classifier == 'xgbTree'){
      n_jobs_ = 1
    } else{
      n_jobs_ = n_jobs
    }
    
    # Load and format data
    data = load_data(
      dataset = dataset,
      criteria = criteria,
      hamd=hamd,
      include.vars = include.vars,
      include.freesurfer = include.freesurfer,
      freesurfer.measures = freesurfer.measures,
      average.lr = average.lr
    )
    
    y = data$group
    
    # Transform factors to dummy variables for classifiers to require it
    if (classifier != c('randomForest')) {
      X = model.matrix(group ~ ., data = data)
      X = X[,-1]  # remove intercept
    } else {
      X = data %>% select(-group)
    }
    
    file.suffix = get_suffix(
      include.vars = include.vars,
      include.freesurfer = include.freesurfer,
      average.lr = average.lr,
      freesurfer.measures = freesurfer.measures
    )
    
    out.rds = file.path('data', dataset, results.dir, 
      paste(classifier, criteria, hamd, file.suffix, 'rds', sep = '.'))  
    
    if (!file.exists(out.rds)) {
      
      cat('Training final models\n')
      print(out.rds)
      
      seeds = 1:n_bootstrap
    
      fits = mcmapply(
        train_classifier,
        list(X), list(y),
        classifier = classifier,
        use.class.weight = use.class.weight,
        bootstrap = T,
        seed = seeds,
        mc.cores = n_jobs_,
        SIMPLIFY = F
      )
      
      saveRDS(fits, file = out.rds)
      
    }
    
  }
}


create_roc = function(df) {
  
  # Create ROC plot of CV results
  
  # Pool data across CV folds
  y = df$y %>% unlist() %>% c()
  prob = df$prob %>% unlist() %>% c()
  
  # Compute ROC
  roc.fit = pROC::roc(
    y, prob,
    levels = c('control', 'case'),
    direction = '<'
  )

  # Interpolate ROC curve to regular interval
  xout = seq(0, 1, by = 0.01)
  interp = stats::approx(
    roc.fit$specificities,
    roc.fit$sensitivities,
    n = 1000
  )
  
  # Store interpolated results
  data.frame(
    specificity = interp$x,
    sensitivity = interp$y
  )
  
}


evaluate_metrics = function(df) {
  
  # Evaluate metrics for repeated CV

  y = df$y %>% unlist() %>% c()
  pred = df$pred %>% unlist() %>% c()
  prob = df$prob %>% unlist() %>% c()
  
  # Specificity & sensitivity
  pred = factor(pred %>% recode(`1`='control', `2`='case'),
                levels = c('control', 'case'))
  df.caret = data.frame(obs = y, pred = pred)
  caret.summary = caret::multiClassSummary(df.caret, lev = c('control', 'case'))
  caret.summary['Sensitivity']
  
  # Evaluate ROC
  roc.fit= pROC::roc(
    y, prob,
    levels = c('control', 'case'),
    direction = '<'
  )

  tibble(
    auc = pROC::auc(roc.fit)[1],
    balanced_accuracy = caret.summary['Balanced_Accuracy'],
    sensitivity = caret.summary['Sensitivity'],
    specificity = caret.summary['Specificity']
  )
  
}


plot_cv_roc_wrapper = function(
    out.roc = NULL,
    cv.rds = NULL,
    permutation.dir = NULL,
    title.prefix = ''
) {
    
  # Read data
  df = readRDS(file = cv.rds)
  
  # Create ROC
  df.roc = bind_rows(df$cv_results %>% lapply(create_roc))
  df.roc$group = 'Not-Permuted'
  
  # Gather classifier metrics across CV     
  metrics = bind_rows(df$cv_results %>% lapply(evaluate_metrics))
  
  auc_mean = mean(metrics$auc)
  accuracy_mean = mean(metrics$balanced_accuracy)
  sensitivity_mean = mean(metrics$sensitivity)
  specificity_mean = mean(metrics$specificity)
  auc_sd = sd(metrics$auc)
  accuracy_sd = sd(metrics$balanced_accuracy)
  sensitivity_sd = sd(metrics$sensitivity)
  specificity_sd = sd(metrics$specificity)
  
  # ROC
  title = bquote(atop(bold(.(title.prefix)),
                      atop(.(sprintf('AUC: %0.2f (%0.2f), Balanced Accuracy: %0.1f%% (%0.1f%%)',
                                     auc_mean, auc_sd,
                                     accuracy_mean*100, accuracy_sd*100)),
                           .(sprintf('Sensitivity: %0.1f%% (%0.1f%%), Specificity: %0.1f%% (%0.1f%%)',
                                     sensitivity_mean*100, sensitivity_sd*100,
                                     specificity_mean*100, specificity_sd*100)))
  ))

  if (permutation.dir != ''){
    
    group = 'group'  # tells the plotting function what the groups are
    
    # Load permuted results
    rds.files = as.list(file.path(permutation.dir, list.files(permutation.dir, pattern = '*rds')))
    perm_cv_results = lapply(rds.files, readRDS)
    
    # Aggregate permuted ROC
    df.perm.roc = bind_rows(lapply(
      perm_cv_results, function(x) bind_rows(x$cv_results %>% lapply(create_roc))
    ))
    df.perm.roc$group = 'Permuted'
    df.roc = rbind(df.roc, df.perm.roc)
    
  } else {
    group = NA
  }
  
  # Plot ROC
  plot_cv_roc(
    df.roc,
    group = group,
    title = title
  )
  
}


plot_cv_roc = function(
  df,
  title = '',
  group = NA,
  x.length = NA,
  plot.sd = T,
  plot.color = F
) {
  
  df$`1-specificity` = 1 - df$specificity
  
  if (is.na(group)) {
    
    df.summary = df %>%
      group_by(`1-specificity`) %>%
      summarise(mean = mean(sensitivity),
                sd_min = mean(sensitivity) - sd(sensitivity),
                sd_max = mean(sensitivity) + sd(sensitivity))
    
    g = ggplot(df.summary, aes(x=`1-specificity`, y=mean))
    
  } else {
    
    df.summary = df %>%
      group_by(`1-specificity`, group) %>%
      summarise(mean = mean(sensitivity),
                sd_min = mean(sensitivity) - sd(sensitivity),
                sd_max = mean(sensitivity) + sd(sensitivity))
  
    g = ggplot(
        df.summary,
        aes(x=`1-specificity`, y=mean, color = group, fill = group)
      )
    
  }
  
  g = g +
    geom_line(size=1, alpha=0.8) +
    geom_abline(intercept = 0, slope = 1, linetype = 2)
    
  if (plot.sd) { 
    g = g + geom_ribbon(aes(ymin=sd_min, ymax=sd_max), alpha=0.2)
  }
  
  g = g +
    coord_fixed() +
    ggtitle(title) +
    xlab('1-Specificity') +
    ylab('Sensitivity') + 
    scale_x_continuous(expand = c(0, 0)) +  # remove white space at edges of axis
    scale_y_continuous(expand = c(0, 0)) +  # remove white space at edges of axis
    theme_pubr(base_size = 10) +
    theme(plot.title = element_text(size = 8))
  
  if (plot.color) {
    
    g = g +
      scale_color_manual(
        name = element_blank(),
        values = c('red', 'blue', 'darkgreen', 'orange')
      ) +
      scale_fill_manual(
        name = element_blank(),
        values = c('red', 'blue', 'darkgreen', 'orange')
      )
    
  } else{

    if (plot.sd) {
      g = g +
        scale_color_grey(start = 0.2, end = 0.8, name = element_blank()) +
        scale_fill_grey(start = 0.2, end = 0.6, name = element_blank()) +
        guides(color = 'none')  # remove color legend
    } else {
      g = g +
        labs(color = element_blank(), fill = element_blank()) +  # remove legend titles
        scale_color_grey(start = 0.2, end = 0.6, name = element_blank())
    }
    
  } 
  
  g
  
}


vbm_data_on_demand = function(
  dataset,
  criteria,
  training_subjects,
  test_subjects,
  out_dir
) {
  
  if (criteria == 'response') {
    criteria_ = 'Responder'
  }
  
  if (criteria == 'remission') {
    criteria_ = 'Remitter'
  }
  
  training_str = paste0('[', paste(paste0('"', training_subjects, '"'), collapse = ';'), ']')
  test_str = paste0('[', paste(paste0('"', test_subjects, '"'), collapse = ';'), ']')
  
  cmd = paste0(
    'matlab -nosplash -r \'addpath("', file.path(getwd(), 'classification'), '");',
    paste0('vbm_export_classification_data("', dataset, '","', criteria_, '",', training_str, ',', test_str, ',"', out_dir, '");'),
    'quit;\''
  ) 
  cat(cmd)
  system(cmd)
  
}