library(dplyr)
load(file='creatinine.rda')
df = creatinine
df = na.omit(df)
write.csv(df, "creatinine.csv", row.names= FALSE)
