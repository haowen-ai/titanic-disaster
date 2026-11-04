# Titanic survival model with logistic regression (glm).
# Reads train.csv and test.csv from src/data, cleans the data, fits the
# model and prints every step to the terminal.

suppressPackageStartupMessages(library(dplyr))
options(width = 120)

seed <- 42
submission_file <- "submission_r.csv"
keep_titles <- c("Mr", "Mrs", "Miss", "Master")
title_map <- c(Mlle = "Miss", Ms = "Miss", Mme = "Mrs")
# Sex is left out: Title already carries most of it (Mr/Master are men,
# Mrs/Miss are women; only the few Rare titles mix both). With both in the
# model the coefficients blow up. Training accuracy is the same without it.
model_formula <- Survived ~ Pclass + Age + Fare + Embarked + Title +
  FamilySize + IsAlone + HasCabin

section <- function(title) cat("\n===", title, "===\n")

# Folder of this script when run with Rscript, so src/data can be found
# from any working directory.
script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) return(getwd())
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
}

load_csv <- function(name, data_dir) {
  path <- file.path(data_dir, name)
  if (!file.exists(path)) {
    stop(path, " not found. Download train.csv and test.csv from Kaggle, ",
         "put them in src/data, then rebuild the image (see README).", call. = FALSE)
  }
  # Empty strings (Cabin, Embarked) count as missing.
  df <- read.csv(path, stringsAsFactors = FALSE, na.strings = c("", "NA"))
  cat(sprintf("Loaded %s: %d rows, %d columns\n", name, nrow(df), ncol(df)))
  df
}

explore <- function(df) {
  cat("Column types:\n")
  print(sapply(df, class))
  cat("\nFirst 5 rows:\n")
  print(head(df, 5))
  cat("\nMissing values per column:\n")
  print(colSums(is.na(df)))
  cat(sprintf("\nSurvival rate overall: %.3f\n", mean(df$Survived)))
  for (col in c("Sex", "Pclass", "Embarked")) {
    cat(sprintf("\nSurvival rate by %s:\n", col))
    rates <- df %>%
      group_by(.data[[col]]) %>%
      summarise(survival_rate = round(mean(Survived), 3), passengers = n(), .groups = "drop")
    print(as.data.frame(rates))
  }
}

extract_title <- function(names) {
  # Names look like "Braund, Mr. Owen Harris": the title sits between ", " and ".".
  raw <- trimws(sub("^[^,]*,\\s*([^.]+)\\..*$", "\\1", names))
  mapped <- ifelse(raw %in% names(title_map), title_map[raw], raw)
  unname(ifelse(mapped %in% keep_titles, mapped, "Rare"))
}

# Fill values come from the training data only and are reused on test.
learn_fill_values <- function(train) {
  by_title <- train %>%
    mutate(Title = extract_title(Name)) %>%
    group_by(Title) %>%
    summarise(age = round(median(Age, na.rm = TRUE), 1), .groups = "drop")
  ports <- table(train$Embarked)
  list(
    age_by_title = setNames(by_title$age, by_title$Title),
    age_overall = round(median(train$Age, na.rm = TRUE), 1),
    embarked = names(ports)[which.max(ports)],
    fare = median(train$Fare, na.rm = TRUE)
  )
}

# Adds features and fills missing values, printing each change.
# Factor levels are fixed so train and test get the same dummy columns.
clean <- function(df, fill, label, verbose = TRUE) {
  say <- function(...) if (verbose) cat(sprintf("[%s] ", label), ..., "\n", sep = "")
  out <- df

  out$Title <- extract_title(out$Name)
  counts <- table(out$Title)
  say("Added Title from Name: ", paste(names(counts), counts, sep = "=", collapse = ", "))

  missing_age <- is.na(out$Age)
  title_age <- unname(fill$age_by_title[out$Title])
  title_age[is.na(title_age)] <- fill$age_overall
  out$Age[missing_age] <- title_age[missing_age]
  say("Filled ", sum(missing_age), " missing Age values with the median age of each title (from train): ",
      paste(names(fill$age_by_title), fill$age_by_title, sep = "=", collapse = ", "))

  missing_port <- sum(is.na(out$Embarked))
  out$Embarked[is.na(out$Embarked)] <- fill$embarked
  say("Filled ", missing_port, " missing Embarked values with '", fill$embarked, "'")

  missing_fare <- sum(is.na(out$Fare))
  out$Fare[is.na(out$Fare)] <- fill$fare
  say("Filled ", missing_fare, " missing Fare values with median ", fill$fare)

  out <- out %>%
    mutate(
      FamilySize = SibSp + Parch + 1,
      IsAlone = as.integer(FamilySize == 1),
      HasCabin = as.integer(!is.na(Cabin)),
      Pclass = factor(Pclass, levels = c(1, 2, 3)),
      Sex = factor(Sex, levels = c("female", "male")),
      Embarked = factor(Embarked, levels = c("C", "Q", "S")),
      Title = factor(Title, levels = c(keep_titles, "Rare"))
    )
  say("Added FamilySize (SibSp + Parch + 1), IsAlone (", sum(out$IsAlone), " alone), HasCabin (",
      sum(out$HasCabin), " with a cabin)")
  say("Turned Pclass, Sex, Embarked and Title into factors (glm makes the dummy columns)")

  dropped <- c("PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch")
  out <- out[, setdiff(names(out), dropped)]
  say("Dropped ", paste(dropped, collapse = ", "), "; now ", nrow(out), " rows, ", ncol(out), " columns")
  say("Missing values left: ", sum(is.na(out[, setdiff(names(out), "Survived")])))
  out
}

accuracy <- function(model, df) {
  predicted <- as.integer(predict(model, newdata = df, type = "response") > 0.5)
  mean(predicted == df$Survived)
}

main <- function() {
  data_dir <- file.path(dirname(script_dir()), "data")

  section("Step 1: load train.csv")
  train <- load_csv("train.csv", data_dir)

  section("Step 2: explore the training data")
  explore(train)

  section("Step 3: clean the data and add features")
  fill <- learn_fill_values(train)
  train_clean <- clean(train, fill, "train")
  cat("Model formula:", deparse(model_formula, width.cutoff = 200), "\n")
  cat("Sex is not in the formula because Title already tells men from women",
      "(except the few Rare titles); keeping both makes the glm coefficients unstable.\n")

  section("Step 4: logistic regression, check on a holdout set")
  # Kaggle's test.csv has no answers, so keep 20% of train.csv aside to see
  # how the model does on passengers it has not seen. Split within each class
  # so both parts keep the same survival rate.
  set.seed(seed)
  holdout_rows <- unlist(lapply(split(seq_len(nrow(train)), train$Survived),
                                function(rows) sample(rows, round(0.2 * length(rows)))))
  part <- train[-holdout_rows, ]
  holdout <- train[holdout_rows, ]
  part_fill <- learn_fill_values(part)
  part_clean <- clean(part, part_fill, "train part", verbose = FALSE)
  holdout_clean <- clean(holdout, part_fill, "holdout", verbose = FALSE)
  model <- glm(model_formula, data = part_clean, family = binomial)
  cat(sprintf("Split train.csv 80/20 (seed %d): %d rows to fit, %d rows held out\n",
              seed, nrow(part), nrow(holdout)))
  cat(sprintf("Accuracy on the 80%% used to fit: %.4f\n", accuracy(model, part_clean)))
  cat(sprintf("Holdout accuracy (20%% not used to fit): %.4f\n", accuracy(model, holdout_clean)))

  section("Step 5: fit on all of train.csv")
  model <- glm(model_formula, data = train_clean, family = binomial)
  cat(sprintf("Training accuracy (all %d rows): %.4f\n", nrow(train_clean), accuracy(model, train_clean)))
  cat("Coefficients (log-odds, positive = more likely to survive):\n")
  print(round(sort(coef(model), decreasing = TRUE), 3))

  section("Step 6: load test.csv and predict")
  test <- load_csv("test.csv", data_dir)
  cat("Missing values per column:\n")
  print(colSums(is.na(test)))
  test_clean <- clean(test, fill, "test")
  predicted <- as.integer(predict(model, newdata = test_clean, type = "response") > 0.5)
  cat(sprintf("Made %d predictions; predicted survivors: %d (%.3f of test passengers)\n",
              length(predicted), sum(predicted), mean(predicted)))
  submission <- data.frame(PassengerId = test$PassengerId, Survived = predicted)
  cat("First 10 predictions:\n")
  print(head(submission, 10), row.names = FALSE)
  # Same format Kaggle asks for: PassengerId,Survived
  write.csv(submission, file.path(data_dir, submission_file), row.names = FALSE, quote = FALSE)
  cat(sprintf("Saved all %d predictions to src/data/%s\n", nrow(submission), submission_file))

  section("Step 7: test accuracy")
  if ("Survived" %in% names(test)) {
    cat(sprintf("Test accuracy: %.4f\n", accuracy(model, test_clean)))
  } else {
    cat("test.csv has no Survived column (Kaggle keeps the answers), so test accuracy",
        "cannot be computed here.\n")
    cat("Use the holdout accuracy from Step 4 as the estimate. The real test score comes from",
        sprintf("uploading src/data/%s to Kaggle (score in README).\n", submission_file))
  }
}

if (sys.nframe() == 0) {
  main()
}
