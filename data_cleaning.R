library(tidyverse)
library(DescTools)


df <- read.csv("/Users/jakub/PG/collaborative_learning/zpb/EMBED_Open_Data/EMBED_OpenData_clinical_reduced.csv")
full_df <- read.csv("/Users/jakub/PG/collaborative_learning/zpb/EMBED_Open_Data/EMBED_OpenData_clinical.csv")
legend <- read.csv("/Users/jakub/PG/collaborative_learning/zpb/EMBED_Open_Data/resources/AWS_Open_Data_Clinical_Legend.csv")

head(df)
df = subset(df, select = -c(X, numfind, procdate_anon, total_L_find, total_R_find, side, bside, cohort_num, first_3_zip, study_date_anon))

df$type <- full_df$type # Copy Source of the of tissue from full df to reduced df
df$type[df$type == ""] <- NA # Fill empty types with NA 

# Get some columns from full df
df$massshape <- full_df$massshape
df$massmargin <- full_df$massmargin
df$massdens <- full_df$massdens
df$calcdistri <- full_df$calcdistri
df$calcfind <- full_df$calcfind


# There are only 371 males in dataset so we are droping their records
# Remove male records from dataset and drop gender column
df <- df[df$GENDER_DESC == "Female",]
df <- subset(df, select=-GENDER_DESC)

# All records that have path severity have also type defined. There are some records with type defined without path_severity
mean(!is.na(df[!is.na(df$path_severity), "type"]))

# Define birads df without 
birads_df <- subset(df, select=-c(path_severity, type))

head(birads_df)

# Values below are mostly NA (over 80%)
sum(is.na(birads_df$massshape)) / dim(birads_df)[1]
sum(is.na(birads_df$massmargin)) / dim(birads_df)[1]
sum(is.na(birads_df$massdens)) / dim(birads_df)[1]
sum(is.na(birads_df$calcdistri)) / dim(birads_df)[1]
sum(is.na(birads_df$calcfind)) / dim(birads_df)[1]

# Fill empty values with NA
birads_df$massdens[birads_df$massdens == ""] <- NA
birads_df$massmargin[birads_df$massmargin == ""] <- NA
birads_df$massshape[birads_df$massshape == ""] <- NA
birads_df$calcdistri[birads_df$calcdistri == ""] <- NA
birads_df$calcfind[birads_df$calcfind == ""] <- NA

# All records have defined empi_anon, acc_anon, asses, RACE_DESC, ETHNIC_GROUP_DESC, MARTIAL_STATUS_DESC and age_at_study

# Handling of tissuedens NA values

length(unique(birads_df[is.na(birads_df$tissueden),]$empi_anon)) # Total number of unique empi_anons that dont have tissueden for some records

# Get all empi_anon which doesnt have even one tissuedens filled
max_tissuedenses <- birads_df %>% 
  group_by(empi_anon) %>% 
  summarise(max_tissuedens = if (all(is.na(tissueden))) NA else max(tissueden, na.rm = TRUE))
sum(is.na(max_tissuedenses$max_tissuedens)) # gives 48 values

empi_anons_without_tissuedens <- max_tissuedenses[is.na(max_tissuedenses$max_tissuedens), 1][['empi_anon']] # Gets empi_anon which dont have tissuden for any record

# Remove records for empi_anon that dont have tissueden for any records
birads_df <- birads_df[!birads_df$empi_anon %in% empi_anons_without_tissuedens,]

# Replace NA tissueden values with tissueden that is closest by date to that of a record with missing value
fill_tissueden <- function(birads_df) {
  missing_rows <- which(is.na(birads_df$tissueden))
  available_rows <- which(!is.na(birads_df$tissueden))
  
  for (i in missing_rows) {
    age_i <- birads_df$age_at_study[i]
    # Find the available row with closest age
    closest_row <- available_rows[which.min(abs(birads_df$age_at_study[available_rows] - age_i))]
    birads_df$tissueden[i] <- birads_df$tissueden[closest_row]
  }
  return(birads_df)
}

# Apply to each patient group
birads_df <- birads_df %>%
  group_by(empi_anon) %>%
  group_modify(~ fill_tissueden(.x)) %>%
  ungroup()

# There are no missing values except mass and calc values
sum(is.na(birads_df[,1:9]))

birads_df <- birads_df[,1:9]

write.csv(birads_df, "EMBED_OpenData_clinical_cleaned.csv")
