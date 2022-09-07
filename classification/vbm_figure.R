source('classification/classification.utils.R')

dir.create('figures', showWarnings = F, recursive = T)

results.dir = 'cv_results'
classifiers = c('glmnet', 'randomForest', 'svmRadialWeights', 'xgbTree')


extract_classifier_roc = function(params) {
  
  dataset = params$dataset
  include.vars = params$include.vars[[1]]
  include.freesurfer = params$include.freesurfer
  criteria = params$criteria
  hamd = params$hamd
  classifier = params$classifier
  include.vbm = params$include.vbm

  file.suffix = get_suffix(
    include.vars = include.vars,
    include.freesurfer = include.freesurfer,
    average.lr = average.lr,
    freesurfer.measures = freesurfer.measures,
    include.vbm = params$include.vbm
  )
  
  cv.rds = file.path('data', dataset, results.dir, 
     paste(classifier, criteria, hamd, file.suffix, 'rds', sep = '.'))   
  
  # Read data
  df = readRDS(file = cv.rds)
  
  # Create ROC
  df.roc = bind_rows(df$cv_results %>% lapply(create_roc))
  df.roc$group = classifier
  
  df.roc
  
}


roc_plot_wrapper  = function(criteria, dataset) {
  
  study.params =
    tibble(include.freesurfer = F) %>%
    mutate(
      dataset = dataset,
      criteria = criteria,
      include.vbm = T,
      include.vars = list(c('age', 'sex', 'hamd6.base', 'single.recurrent')),
      hamd = 'hamd6'
    ) %>%
    full_join(tibble(classifier = classifiers), by = character()) %>%
    group_split(row_number(), .keep = FALSE)

  df.roc =
    lapply(study.params, extract_classifier_roc) %>%
    bind_rows() %>%
    mutate(group =
      recode(group, 
        glmnet = 'Elastic Net',
        svmRadialWeights = 'SVM',
        randomForest = 'Random Forest',
        xgbTree = 'Boosted Trees'
      )
    )
  
  # Plot
  plot_cv_roc(
    df.roc,
    title = str_to_title(criteria),
    group = 'group',
    plot.sd = F,
    plot.color = T
  )
  
}


plot.list = lapply(c('response', 'remission'), roc_plot_wrapper, 'NP1')

ggarrange(
  plotlist = plot.list,
  ncol = 2,
  common.legend = T,
  labels='AUTO',
  legend = 'bottom'
)

ggsave(
  file = file.path('figures', 'Fig_S2.png'),
  width=180, height=100,
  dpi=600,
  units = 'mm',
  bg = 'white'
)