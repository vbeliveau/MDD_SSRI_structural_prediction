library(broom)
library(flextable)

source('classification/classification.utils.R')

dir.create('tables', showWarnings = F, recursive = T)


lm.stats.table = function(
  dataset = 'NP1',
  criteria = 'response',
  hamd = 'hamd6',
  all.digits = F
) { 

  
  # Load and format data
  data = load_data(
    dataset,
    criteria = criteria,
    hamd=hamd,
    include.freesurfer = T,
    include.vars = c('age', 'sex'),
    freesurfer.measures = c('thickness', 'volume'),
    average.lr = F
  )

  # Transform to long format
  regions = data %>% select(-group, -icv.volume, -age, -sex) %>% colnames
  data = data %>%
    pivot_longer(all_of(regions), names_to = 'region') %>%
    mutate(measure = ifelse(endsWith(region, '.thickness'), 'thickness', 'volume')) %>%
    group_by(group, measure)
  
  # Perform regression
  lm.stats = rbind(
      data %>%
        filter(measure == 'thickness') %>%
        group_by(region) %>%
        do(tidy(lm(value ~ group + age + sex, data = .))) %>%
        filter(term == 'groupcase') %>% select(-term),
      data %>%
        filter(measure == 'volume') %>%
        group_by(region) %>%
        do(tidy(lm(value ~ group + age + sex + icv.volume, data = .))) %>%
        filter(term == 'groupcase') %>% select(-term)
  )

  # Compute Cohen's d according to No Alterations of Brain Structural Asymmetry in Major Depressive Disorder: An ENIGMA Consortium Analysis
  # Cohen's d: t*sqrt(1/n1 + 1/n2)
  
  n1 = sum(data$group == 'control')
  n2 = sum(data$group == 'case')
  lm.stats$`Cohen's d` = lm.stats$statistic*sqrt((1/n1)+(1/n2))
  lm.stats = lm.stats %>% relocate(`Cohen's d`, .after=region)
  
  # Correct p-values for the number of comparisons
  lm.stats$`FDR p-value` = p.adjust(lm.stats$p.value, method='fdr')
  
  # Clean up region names
  lm.stats$region = unlist(lapply(lm.stats$region, clean_name))
  
  # Print full significant results
  print(lm.stats %>% filter(`FDR p-value` < 0.05))
  
  # Rename columns and remove unwanted
  lm.stats = lm.stats %>%
    rename(Region=region, `p-value`=p.value) %>%
    select(-c(estimate, std.error, statistic))
  
  # Create table
  if (all.digits) {
    tab = lm.stats %>%
      flextable() %>%
      flextable::save_as_docx( path = out.docx )
  } else {
    tab = lm.stats %>% flextable() %>%
      colformat_double(j = colnames(lm.stats)[2], digits = 3) %>%
      colformat_double(j = colnames(lm.stats)[3:ncol(lm.stats)], digits = 2)
  }

  tab 

}


for (dataset in c('NP1', 'EMBARC')) {
  for (criteria in c('response', 'remission')) {
    
    print(paste(dataset, criteria))
    
    out.docx = file.path('tables',
       paste('regional_analysis', dataset, criteria, 'docx', sep = '.'))
    
    lm.stats.table(
      dataset = dataset,
      criteria = criteria,
      hamd = 'hamd6'
    ) %>%
    flextable::save_as_docx(path = out.docx)
    
  }
}
