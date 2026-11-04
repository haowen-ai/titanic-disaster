# Installs the R packages used by titanic.R at fixed versions.
# The repo is a dated snapshot of CRAN, so every rebuild gets the same
# versions (dependencies included). Linux binaries make the install fast.

snapshot_date <- "2026-11-03"
packages <- c(dplyr = "1.2.1")

# Use the binary repo for the Ubuntu release the base image is built on.
os_release <- readLines("/etc/os-release")
codename <- sub("^VERSION_CODENAME=", "", grep("^VERSION_CODENAME=", os_release, value = TRUE))
repo <- sprintf("https://p3m.dev/cran/__linux__/%s/%s", codename, snapshot_date)
cat("Installing from", repo, "\n")

install.packages(names(packages), repos = repo)

for (pkg in names(packages)) {
  installed <- as.character(packageVersion(pkg))
  cat(sprintf("%s: wanted %s, installed %s\n", pkg, packages[[pkg]], installed))
  if (installed != packages[[pkg]]) {
    stop("Version mismatch for ", pkg, call. = FALSE)
  }
}
